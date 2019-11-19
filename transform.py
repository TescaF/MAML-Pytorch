from tf import transformations as tr
from copy import deepcopy
import os
import rosbag
import math
import pdb
import  numpy as np
import  argparse


def main():
    demo_num = 1
    tf_path = os.path.expanduser("~") + '/data/bags/' + args.src + "_" + str(demo_num) + ".bag"
    while os.path.exists(tf_path):
        print("-----Loading source demo " + str(demo_num) + "-----")
        tf_bag = rosbag.Bag(tf_path)
        kf_list, msgs = [],[]
        #for topic, msg, t in tf_bag.read_messages(topics=['eef_pose_relative']):
        for topic, msg, t in tf_bag.read_messages():
            if topic == 'eef_pose_relative':
                kf_list.append([msg,t])
            else:
                msgs.append([topic, msg, t])
        
        print("-----Loading transforms-----")
        tf_file = os.path.expanduser("~") + "/data/output/src_" + args.src + "-demo_" + str(demo_num) + "-tgt_" + args.tgt + ".txt"
        with open(tf_file, 'r') as in_file:
            lines = [line.strip() for line in in_file]
        tfs,tf_poses = [],[]

        ## Get TF to apply to keyframes
        for kf_i in range(len(lines)/3):
            data = lines[(kf_i * 3) + 1]
            median = [float(m) for m in data.split("] [")[1].split()]
            tfs.append(median)
        comp_pt = kf_list[args.kf][0].pose.position
        diff_x = tfs[args.kf][0] - comp_pt.x
        diff_y = tfs[args.kf][1] - comp_pt.y

        print("-----Writing bags-----")
        out_path = os.path.expanduser("~") + '/data/bags/tfd/src_' + args.src + '_demo' + str(demo_num) + '_kf' + str(args.kf) + '_to_tgt_' + args.tgt + '.bag'
        with rosbag.Bag(out_path, 'w') as out_bag:
            for kf in kf_list:
                kf[0].pose.position.x += median[0]
                kf[0].pose.position.y += median[1]
                trans = kf[0].pose.position
                rot = kf[0].pose.orientation
                inv = tr.concatenate_matrices(tr.translation_matrix((trans.x,trans.y,trans.z)), tr.quaternion_matrix((rot.x,rot.y,rot.z,rot.w)))
                inv = tr.inverse_matrix(inv)
                trans_i = tr.translation_from_matrix(inv)
                rot_i = tr.quaternion_from_matrix(inv)
                pi = deepcopy(kf[0])
                pi.pose.position.x = trans_i[0]
                pi.pose.position.y = trans_i[1]
                pi.pose.position.z = trans_i[2]
                pi.pose.orientation.x = rot_i[0]
                pi.pose.orientation.y = rot_i[1]
                pi.pose.orientation.z = rot_i[2]
                pi.pose.orientation.w = rot_i[3]
                pi.header.frame_id = "goal"
                out_bag.write('eef_pose_relative', pi, kf[1])
            for m in msgs:
                out_bag.write(m[0], m[1], m[2])

        demo_num += 1
        tf_path = os.path.expanduser("~") + '/data/bags/' + args.src + "_" + str(demo_num) + ".bag"

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--src', type=str, help='epoch number', default="")
    argparser.add_argument('--tgt', type=str, help='epoch number', default="")
    argparser.add_argument('--kf', type=int, help='epoch number', default=10)

    args = argparser.parse_args()

    main()
