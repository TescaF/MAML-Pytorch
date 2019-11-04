#!/usr/bin/env python
import rospy

# to get commandline arguments
import sys

# because of transformations
import tf

import tf2_ros
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('goal_tf_broadcaster')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "j2s7s300_link_base"
    static_transformStamped.child_frame_id = "goal"

    pose_tf = [0.075, 0.52, 0.06, -0.1, -0.69, -0.28, -0.65]
    static_transformStamped.transform.translation.x = pose_tf[0]
    static_transformStamped.transform.translation.y = pose_tf[1]
    static_transformStamped.transform.translation.z = pose_tf[2]
    static_transformStamped.transform.rotation.x = 0
    static_transformStamped.transform.rotation.y = 0
    static_transformStamped.transform.rotation.z = 0
    static_transformStamped.transform.rotation.w = 1

    broadcaster.sendTransform(static_transformStamped)
    rospy.spin()
