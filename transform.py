from tf import transformations as tr
from copy import deepcopy
import os
import rosbag
import math
import pdb
import  numpy as np
import  argparse

goal_tf = [0.075, 0.52, 0.06]

def base_to_ee(pt):
    trans = pt.pose.position
    rot = pt.pose.orientation
    inv = tr.concatenate_matrices(tr.translation_matrix((trans.x-goal_tf[0],trans.y-goal_tf[1],trans.z-goal_tf[2])), tr.quaternion_matrix((rot.x,rot.y,rot.z,rot.w)))
    inv = tr.inverse_matrix(inv)
    trans_i = tr.translation_from_matrix(inv)
    rot_i = tr.quaternion_from_matrix(inv)

    pi = deepcopy(pt)
    pi.pose.position.x = trans_i[0]
    pi.pose.position.y = trans_i[1]
    pi.pose.position.z = trans_i[2]
    pi.pose.orientation.x = rot_i[0]
    pi.pose.orientation.y = rot_i[1]
    pi.pose.orientation.z = rot_i[2]
    pi.pose.orientation.w = rot_i[3]
    pi.header.frame_id = "j2s7s300_link_base"

    return pi

def main():
    demo_num = 1
    tf_path = os.path.expanduser("~") + '/test_bagfiles/' + args.src + "_t" + str(args.t) + "_" + str(demo_num) + ".bag"
    while os.path.exists(tf_path):
        print("-----Loading source demo " + str(demo_num) + "-----")
        tf_bag = rosbag.Bag(tf_path)
        kf_list, msgs = [],[]
        #for topic, msg, t in tf_bag.read_messages(topics=['eef_pose_relative']):
        for topic, msg, t in tf_bag.read_messages():
            if topic == 'eef_pose_relative':
                kf_list.append([base_to_ee(msg),t])
            elif topic != "eef_pose_j2s7s300_link_base":
                msgs.append([topic, msg, t])
        
        print("-----Loading transforms-----")
        tf_file = os.path.expanduser("~") + "/data/output/src_" + args.src + "-demo_" + str(demo_num) + "-tgt_" + args.tgt + "-t_" + str(args.t) + ".txt"
        with open(tf_file, 'r') as in_file:
            lines = [line.strip() for line in in_file]
        tfs,tf_poses,losses,diffs = [],[],[],[]

        ## Get TF to apply to keyframes
        for kf_i in range(len(lines)/3):
            data = lines[(kf_i * 3) + 1]
            loss = float(data.split("] [")[0].split("[")[1])
            median = [float(m) for m in data.split("] [")[1].split()]
            losses.append(loss ** 2.0)
            tfs.append(median)
            comp_pt = kf_list[kf_i][0].pose.position
            diff_x = median[0] - comp_pt.x
            diff_y = median[1] + comp_pt.z
            diffs.append(np.array([diff_x,diff_y]))
        loss_norm = np.sum(1.0/np.array(losses[1:]))
        loss_w = 1.0 / (np.array(losses[1:]) * loss_norm)
        w_diffs = np.multiply(np.stack(diffs[1:]).transpose(), loss_w)
        diff_x, diff_y = w_diffs.transpose().sum(axis=0).tolist()
        pdb.set_trace()
        #comp_pt = base_to_ee(kf_list[args.kf][0]).pose.position
        print("Diff: " + str([diff_x,diff_y]))
        print("-----Writing bags-----")
        out_path = os.path.expanduser("~") + '/data/bags/tfd/src_' + args.src + '_demo' + str(demo_num) + '_kf' + str(args.kf) + '_to_tgt_' + args.tgt + '_t' + str(args.t) + '.bag'
        with rosbag.Bag(out_path, 'w') as out_bag:
            for kf_i in range(len(kf_list)):
                trans = kf_list[kf_i][0].pose.position
                rot = kf_list[kf_i][0].pose.orientation
                #tmp_diff_x = tfs[kf_i][0] - trans.x
                #tmp_diff_y = tfs[kf_i][1] - trans.z
                #print("Diff: " + str(tmp_diff_x) + ", " + str(tmp_diff_y))

                #inv = tr.concatenate_matrices(tr.translation_matrix((-0.009,-0.304,-0.032)), tr.quaternion_matrix((-0.512,-0.383,-0.468,0.61)))
                inv = tr.concatenate_matrices(tr.translation_matrix((trans.x+diff_x,trans.y,trans.z+diff_y)), tr.quaternion_matrix((rot.x,rot.y,rot.z,rot.w)))
                inv = tr.inverse_matrix(inv)
                trans_i = tr.translation_from_matrix(inv)
                rot_i = tr.quaternion_from_matrix(inv)
                pi = deepcopy(kf_list[kf_i][0])
                pi.pose.position.x = trans_i[0] + goal_tf[0]
                pi.pose.position.y = trans_i[1] + goal_tf[1]
                pi.pose.position.z = trans_i[2] + goal_tf[2]
                pi.pose.orientation.x = rot_i[0]
                pi.pose.orientation.y = rot_i[1]
                pi.pose.orientation.z = rot_i[2]
                pi.pose.orientation.w = rot_i[3]
                pi.header.frame_id = "j2s7s300_link_base"
                out_bag.write('eef_pose_j2s7s300_link_base', pi, kf_list[kf_i][1])
                out_bag.write('eef_pose_relative', pi, kf_list[kf_i][1])
            for m in msgs:
                out_bag.write(m[0], m[1], m[2])

        demo_num += 1
        tf_path = os.path.expanduser("~") + '/test_bagfiles/' + args.src + "_t" + str(args.t) + "_" + str(demo_num) + ".bag"

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--src', type=str, help='epoch number', default="")
    argparser.add_argument('--tgt', type=str, help='epoch number', default="")
    argparser.add_argument('--kf', type=int, help='epoch number', default=10)
    argparser.add_argument('--t', type=int, help='epoch number', default=10)

    args = argparser.parse_args()

    main()
