import os
import cv2
import sys
import json
import re
import random
import traceback

import numpy as np
from collections import Counter

import alfworld.gen
import alfworld.gen.constants as constants
from alfworld.agents.controller.base import BaseAgent
from alfworld.agents.utils.misc import extract_admissible_commands_with_heuristics


class OracleAgent(BaseAgent):

    def __init__(self, env, traj_data, traj_root,
                 load_receps=False, debug=False,
                 goal_desc_human_anns_prob=0.0,
                 use_gt_relations=False):
        self.openable_points = self.get_openable_points(traj_data)
        self.use_gt_relations = use_gt_relations
        self.exploration_frames = []
        super().__init__(env, traj_data, traj_root,
                         load_receps=load_receps, debug=debug,
                         goal_desc_human_anns_prob=goal_desc_human_anns_prob)

    def get_openable_points(self, traj_data):
        scene_num = traj_data['scene']['scene_num']
        openable_json_file = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%d-openable.json' % scene_num)
        with open(openable_json_file, 'r') as f:
            openable_points = json.load(f)
        return openable_points

    def get_obj_cls_from_metadata(self, name):
        objs = [obj for obj in self.env.last_event.metadata['objects'] if obj['visible'] and name in obj['objectType']]
        return objs[0] if len(objs) > 0 else None

    def get_obj_id_from_metadata(self, object_id):
        objs = [obj for obj in self.env.last_event.metadata['objects'] if object_id == obj['objectId']]
        return objs[0] if len(objs) > 0 else None

    def get_num_interactable_objs(self, recep_id):
        return len([obj for obj in self.env.last_event.metadata['objects'] if obj['visible'] and obj['parentReceptacles'] and recep_id in obj['parentReceptacles']])

    def get_exploration_frames(self):
        return self.exploration_frames

    # use pre-computed openable points from ALFRED to store receptacle locations
    def explore_scene(self):
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        for object_id, point in self.openable_points.items():
            action = {'action': 'TeleportFull',
                      'x': point[0],
                      'y': agent_height,
                      'z': point[1],
                      'rotateOnTeleport': False,
                      'rotation': point[2],
                      'horizon': point[3]}
            event = self.env.step(action)

            if event.metadata['lastActionSuccess']:
                self.exploration_frames.append(np.array(self.env.last_event.frame[:,:,::-1]))
                instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
                color_to_object_id = self.env.last_event.color_to_object_id

                # find unique instance segs
                color_count = Counter()
                for x in range(instance_segs.shape[0]):
                    for y in range(instance_segs.shape[1]):
                        color = instance_segs[y, x]
                        color_count[tuple(color)] += 1

                for color, num_pixels in color_count.most_common():
                    if color in color_to_object_id:
                        object_id = color_to_object_id[color]
                        object_type = object_id.split('|')[0]
                        if "Basin" in object_id:
                            object_type += "Basin"

                        if object_type in self.STATIC_RECEPTACLES:
                            if object_id not in self.receptacles:
                                self.receptacles[object_id] = {
                                    'object_id': object_id,
                                    'object_type': object_type,
                                    'locs': action,
                                    'num_pixels': num_pixels,
                                    'num_id': "%s %d" % (object_type.lower(), self.get_next_num_id(object_type, self.receptacles)),
                                    'closed': True if object_type in constants.OPENABLE_CLASS_LIST else None
                                }
                            elif object_id in self.receptacles and num_pixels > self.receptacles[object_id]['num_pixels']:
                                self.receptacles[object_id]['locs'] = action  # .append(action)
                                self.receptacles[object_id]['num_pixels'] = num_pixels

        # self.save_receps()

    # ground-truth instance segemetations (with consistent object IDs) from THOR
    def get_instance_seg(self):
        instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
        # plt.imsave('output.png',instance_segs)
        inst_color_to_object_id = self.env.last_event.color_to_object_id

        # find unique instance segs
        inst_color_count = Counter()
        for x in range(instance_segs.shape[0]):
            for y in range(instance_segs.shape[1]):
                color = instance_segs[y, x]
                inst_color_count[tuple(color)] += 1
        return inst_color_count, inst_color_to_object_id
    

    # ground-truth object state info maintained by ThorEnv
    def get_object_state(self, object_id):
        is_clean = object_id in self.env.cleaned_objects
        is_hot = object_id in self.env.heated_objects
        is_cool = object_id in self.env.cooled_objects
        is_sliced = 'Sliced' in object_id
        return is_clean, is_cool, is_hot, is_sliced

    def get_admissible_commands(self):
        return extract_admissible_commands_with_heuristics(self.intro, self.frame_desc, self.feedback,
                                                           self.curr_recep, self.inventory)

    def print_frame(self, recep, loc):
        # Modified by Xiaofeng Lin
        # added one more condition to check if the object is visible

        inst_color_count, inst_color_to_object_id = self.get_instance_seg()
        recep_object_id = recep['object_id']

        # for each unique seg add to object dictionary if it's more visible than before
        visible_objects = []
        for color, num_pixels in inst_color_count.most_common():
            if color in inst_color_to_object_id:
                object_id = inst_color_to_object_id[color]
                object_type = object_id.split("|")[0]
                object_metadata = self.get_obj_id_from_metadata(object_id)
                is_obj_in_recep = (object_metadata and object_metadata['parentReceptacles'] and len(object_metadata['parentReceptacles']) > 0 and recep_object_id in object_metadata['parentReceptacles'])
                if object_type in self.OBJECTS and object_metadata and (not self.use_gt_relations or is_obj_in_recep) and object_metadata['visible']:
                    if object_id not in self.objects:
                        # Create base object dictionary
                        obj_name = object_type.lower() if "Sliced" not in object_id else "sliced-%s" % object_type.lower()
                        self.objects[object_id] = {
                            'object_id': object_id,
                            'object_type': object_type,
                            'loc': loc,
                            'num_pixels': num_pixels,
                            'num_id': "%s %d" % (obj_name, self.get_next_num_id(object_type, self.objects))
                        }

                    elif object_id in self.objects and num_pixels > self.objects[object_id]['num_pixels']:
                        self.objects[object_id]['loc'] = loc
                        self.objects[object_id]['num_pixels'] = num_pixels
                    # Add parent if object is in receptacle
                    if is_obj_in_recep:
                        self.objects[object_id]['parent'] = recep['object_id']

                    if self.objects[object_id]['num_id'] not in self.inventory and is_obj_in_recep:
                        visible_objects.append(self.objects[object_id]['num_id'])

        visible_objects_with_articles = ["a %s," % vo for vo in visible_objects]
        feedback = ""
        if len(visible_objects) > 0:
            feedback = "On the %s, you see %s" % (recep['num_id'], self.fix_and_comma_in_the_end(' '.join(visible_objects_with_articles)))
        elif not recep['closed'] and len(visible_objects) == 0:
            feedback = "On the %s, you see nothing." % (recep['num_id'])

        return visible_objects, feedback
    

    def print_frame_desc_oracle(self, action_str):
        # Author: Xiaofeng Lin
        # iterate through all receptacles.
        visible_recep = []
        visible_obj = []
        inst_color_count, inst_color_to_object_id = self.get_instance_seg()

        # Collect visible receptacles and objects
        for color, _ in inst_color_count.most_common():
            if color in inst_color_to_object_id:
                object_id = inst_color_to_object_id[color]
                obj_metadata = self.get_obj_id_from_metadata(object_id)
                if obj_metadata and obj_metadata['visible']:
                    if object_id in self.receptacles:
                        receptacle_name = self.receptacles[object_id]["num_id"]
                        visible_recep.append((object_id, receptacle_name))
                    elif object_id in self.objects:
                        obj_name = self.objects[object_id]["num_id"]
                        visible_obj.append((object_id, obj_name))

        # Map objects to their parent receptacles
        receptacle_contents = {receptacle_name: [] for _, receptacle_name in visible_recep}
        for recep_id, recep_name in visible_recep:
            recep_metadata = self.get_obj_id_from_metadata(recep_id)
            if recep_metadata and recep_metadata["receptacleObjectIds"] is not None:
                for object_id in recep_metadata["receptacleObjectIds"]:
                    if object_id in self.objects:
                        obj_name = self.objects[object_id]["num_id"]
                        if (object_id, obj_name) in visible_obj:
                            receptacle_contents[recep_name].append("a %s," % obj_name)
        
        feedback = ""
        # check inventory objects
        # feedback = ""
        # if len(self.inventory) > 0:
        #     feedback += "You are holding a %s. " % self.inventory[0]
        # else:
        #     feedback += "You are not holding anything. "

        # constructing action feedback
        # action_feedback is "Nothing happens." if the action is failed or there is no visibly changes in the environment.
        # action_feedback is the feedback of the action if there is visibly changes in the environment.
        action_feedback = ""
        cmd = self.parse_command(action_str)
        if cmd['action'] == self.Action.GOTO:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            if recep and (recep['object_id'], recep["num_id"]) in visible_recep:
                action_feedback = "You arrive at %s. " % recep["num_id"]
                action_feedback += self.generate_recep_feedback(recep['object_id'], recep["num_id"], receptacle_contents)
                # delete the recep from visible_recep
                visible_recep.remove((recep['object_id'], recep["num_id"]))
            else:
                action_feedback = "Nothing happens. "
        elif cmd['action'] == self.Action.OPEN:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            if (recep['object_id'], recep["num_id"]) in visible_recep and "You open" in self.feedback:
                action_feedback = "You open the %s. " % recep["num_id"]
                action_feedback += self.generate_recep_feedback(recep['object_id'], recep["num_id"], receptacle_contents)
                # delete the recep from visible_recep
                visible_recep.remove((recep['object_id'], recep["num_id"]))
            else:
                action_feedback = "Nothing happens. "
        elif cmd['action'] == self.Action.CLOSE:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            if (recep['object_id'], recep["num_id"]) in visible_recep and "You close" in self.feedback:
                action_feedback = "You close the %s. " % recep["num_id"]
                # delete the recep from visible_recep
                visible_recep.remove((recep['object_id'], recep["num_id"]))
            else:
                action_feedback = "Nothing happens. "
        
        elif cmd['action'] == self.Action.PUT:
            action_feedback = self.feedback
        elif cmd['action'] == self.Action.PICK:
            action_feedback = self.feedback
        elif cmd['action'] == self.Action.TOGGLE:
            action_feedback = self.feedback
        # for heat, clean, there are no visibly changes in the environment, but vlm is supposed to know based on inventory information
        elif cmd['action'] == self.Action.HEAT:
            action_feedback = self.feedback
        elif cmd['action'] == self.Action.CLEAN:
            action_feedback = self.feedback
        elif cmd['action'] == self.Action.COOL:
            action_feedback = self.feedback
        # for examine, look, inventory, there are no visibly changes in the environment
        elif cmd['action'] == self.Action.EXAMINE or cmd['action'] == self.Action.LOOK or cmd['action'] == self.Action.INVENTORY:
            action_feedback = "Nothing happens. "
        else:
            action_feedback = "Nothing happens. " 


        # for recep_id, recep_name in visible_recep:
        for recep_id, recep_name in visible_recep:
            feedback += self.generate_recep_feedback(recep_id, recep_name, receptacle_contents)
        
        return action_feedback + feedback
                    
    def print_inventory(self):
        # Author: Xiaofeng Lin
        feedback = ""
        if len(self.inventory) > 0:
            object = self.get_object(self.inventory[0], self.objects)
            feedback += self.print_object(object)
            feedback = feedback.replace("This is a normal", "You are holding")
        else:
            feedback += "You are not holding anything. "
        return feedback
    

    def step(self, action_str):
        event = None
        self.feedback = "Nothing happens."

        try:
            cmd = self.parse_command(action_str)

            if cmd['action'] == self.Action.GOTO:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                if recep and recep['num_id'] == self.curr_recep:
                    return self.feedback
                self.curr_loc = recep['locs']
                event = self.navigate(self.curr_loc)
                self.curr_recep = recep['num_id']
                self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)

                # feedback conditions
                loc_id = list(self.receptacles.keys()).index(recep['object_id'])
                loc_feedback = "You arrive at loc %s. " % loc_id
                state_feedback = "The {} is {}. ".format(self.curr_recep, "closed" if recep['closed'] else "open") if recep['closed'] is not None else ""
                loc_state_feedback = loc_feedback + state_feedback
                self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
                self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.PICK:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                if obj in self.visible_objects:
                    object = self.get_object(obj, self.objects)
                    event = self.env.step({'action': "PickupObject",
                                           'objectId': object['object_id'],
                                           'forceAction': True})

                    if event.metadata['lastActionSuccess']:
                        self.inventory.append(object['num_id'])
                        self.feedback = "You pick up the %s from the %s." % (obj, tar)

            elif cmd['action'] == self.Action.PUT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                recep = self.get_object(tar, self.receptacles)
                event = self.env.step({'action': "PutObject",
                                       'objectId': self.env.last_event.metadata['inventoryObjects'][0]['objectId'],
                                       'receptacleObjectId': recep['object_id'],
                                       'forceAction': True})
                if event.metadata['lastActionSuccess']:
                    self.inventory.pop()
                    self.feedback = "You put the %s %s the %s." % (obj, rel, tar)

            elif cmd['action'] == self.Action.OPEN:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                event = self.env.step({'action': "OpenObject",
                                       'objectId': recep['object_id'],
                                       'forceAction': True})
                self.receptacles[recep['object_id']]['closed'] = False
                self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)
                action_feedback = "You open the %s. The %s is open. " % (target, target)
                self.feedback = action_feedback + self.feedback.replace("On the %s" % target, "In it")
                self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.CLOSE:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                event = self.env.step({'action': "CloseObject",
                                       'objectId': recep['object_id'],
                                       'forceAction': True})
                self.receptacles[recep['object_id']]['closed'] = True
                self.feedback = "You close the %s." % target

            elif cmd['action'] == self.Action.TOGGLE:
                target = cmd['tar']
                obj = self.get_object(target, self.objects)
                event = self.env.step({'action': "ToggleObjectOn",
                                       'objectId': obj['object_id'],
                                       'forceAction': True})
                self.feedback = "You turn on the %s." % target

            elif cmd['action'] == self.Action.HEAT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
                recep = self.get_object(tar, self.receptacles)

                # open the microwave, heat the object, take the object, close the microwave
                events = []
                events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PutObject', 'objectId': obj_id, 'receptacleObjectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': obj_id, 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You heat the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.CLEAN:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.env.last_event.metadata['inventoryObjects'][0]
                sink = self.get_obj_cls_from_metadata('BathtubBasin' if "bathtubbasin" in tar else "SinkBasin")
                faucet = self.get_obj_cls_from_metadata('Faucet')

                # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
                events = []
                events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': sink['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': faucet['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': faucet['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You clean the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.COOL:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.env.last_event.metadata['inventoryObjects'][0]
                fridge = self.get_obj_cls_from_metadata('Fridge')

                # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
                events = []
                events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You cool the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.SLICE:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.get_object(obj, self.objects)
                inventory_objects = self.env.last_event.metadata['inventoryObjects']
                if 'Knife' in inventory_objects[0]['objectType']:
                    event = self.env.step({'action': "SliceObject",
                                           'objectId': object['object_id']})
                self.feedback = "You slice %s with the %s" % (obj, tar)

            elif cmd['action'] == self.Action.INVENTORY:
                if len(self.inventory) > 0:
                    self.feedback = "You are carrying: a %s" % (self.inventory[0])
                else:
                    self.feedback = "You are not carrying anything."

            elif cmd['action'] == self.Action.EXAMINE:
                target = cmd['tar']
                receptacle = self.get_object(target, self.receptacles)
                object = self.get_object(target, self.objects)

                if receptacle:
                    self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                    self.frame_desc = str(self.feedback)
                elif object:
                    self.feedback = self.print_object(object)

            elif cmd['action'] == self.Action.LOOK:
                if self.curr_recep == "nothing":
                    self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
                else:
                    self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep

        except:
            if self.debug:
                print(traceback.format_exc())

        if event and not event.metadata['lastActionSuccess']:
            self.feedback = "Nothing happens."
            if self.debug:
                print(event.metadata['errorMessage'])

        if self.debug:
            print(self.feedback)
        return self.feedback

    def generate_recep_feedback(self, recep_id, recep_name, receptacle_contents):
        # Author: Xiaofeng Lin
        """
        Generates feedback based on the receptacle's state and its contents.

        Args:
            recep_id (str): The unique identifier of the receptacle.
            recep_name (str): The name of the receptacle.
            receptacle_contents (dict): A dictionary mapping receptacle names to lists of visible objects.

        Returns:
            str: A feedback string describing the state of the receptacle and its contents.
        """
        feedback = ""
        recep_metadata = self.get_obj_id_from_metadata(recep_id)

        if recep_metadata and "openable" in recep_metadata:
            if recep_metadata["openable"]:
                if recep_metadata.get("isOpen", False):
                    feedback += f"The {recep_name} is open. "
                    if recep_name in receptacle_contents and len(receptacle_contents[recep_name]) > 0:
                        objects_description = self.fix_and_comma_in_the_end(' '.join(receptacle_contents[recep_name]))
                        feedback += f"In the {recep_name}, you see {objects_description} "
                    else:
                        feedback += f"In the {recep_name}, you see nothing. "
                else:
                    feedback += f"The {recep_name} is closed. "
            else:
                if recep_name in receptacle_contents and len(receptacle_contents[recep_name]) > 0:
                    objects_description = self.fix_and_comma_in_the_end(' '.join(receptacle_contents[recep_name]))
                    feedback += f"On the {recep_name}, you see {objects_description} "
                else:
                    feedback += f"On the {recep_name}, you see nothing. "

        return feedback

