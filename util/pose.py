

class Pose:
    """
    Concatenated pose: Body + Foot + Racket
    """
    Nose = 0
    Neck = 1
    Head = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16
    LToe = 17
    RToe = 18
    LHeel = 19
    RHeel = 20

    # Optional keypoints
    RacketHead = 21
    RacketTail = 22

    # Number of keypoints without racket
    NumKeypoints = 21
    NumKeypointsWithRacket = 23

    FlipIndices = [
        0, 1, 2,                # Head
        4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,   # Body
        18, 17, 20, 19,         # Feet
        21, 22]                 # Racket


TORSO_IDXS = [Pose.LShoulder, Pose.RShoulder, Pose.LHip, Pose.RHip]


class Bone:
    """
    Bones for concatenated pose
    """
    # No left right orientation
    Nose2Neck = 0
    Head2Nose = 1
    LShoulder2RShoulder = 2
    LHip2RHip = 3

    # Has left right orientation
    LEar2Nose = 4
    REar2Nose = 5
    LShoulder2Neck = 6
    RShoulder2Neck = 7
    LElbow2LShoulder = 8
    RElbow2RShoulder = 9
    LWrist2LElbow = 10
    RWrist2RElbow = 11
    LHip2LShoulder = 12
    RHip2RShoulder = 13
    LKnee2LHip = 14
    RKnee2RHip = 15
    LAnkle2LKnee = 16
    RAnkle2RKnee = 17
    LToe2LAnkle = 18
    RToe2RAnkle = 19
    LHeel2LAnkle = 20
    RHeel2RAnkle = 21

    # Bones for optional keypoints
    RacketHead2Tail = 22

    NumBones = 22
    NumBonesWithRacket = 23

    KeypointPairs = (
        (Pose.Nose, Pose.Neck),
        (Pose.Head, Pose.Nose),
        (Pose.LShoulder, Pose.RShoulder),
        (Pose.LHip, Pose.RHip),
        (Pose.LEar, Pose.Nose),
        (Pose.REar, Pose.Nose),
        (Pose.LShoulder, Pose.Neck),
        (Pose.RShoulder, Pose.Neck),
        (Pose.LElbow, Pose.LShoulder),
        (Pose.RElbow, Pose.RShoulder),
        (Pose.LWrist, Pose.LElbow),
        (Pose.RWrist, Pose.RElbow),
        (Pose.LHip, Pose.LShoulder),
        (Pose.RHip, Pose.RShoulder),
        (Pose.LKnee, Pose.LHip),
        (Pose.RKnee, Pose.RHip),
        (Pose.LAnkle, Pose.LKnee),
        (Pose.RAnkle, Pose.RKnee),
        (Pose.LToe, Pose.LAnkle),
        (Pose.RToe, Pose.RAnkle),
        (Pose.LHeel, Pose.LAnkle),
        (Pose.RHeel, Pose.RAnkle),
        (Pose.RacketHead, Pose.RacketTail))


assert len(Bone.KeypointPairs) == Bone.NumBonesWithRacket


class Bone_Symmetric:
    """
    Bones with symmetries removed
    """
    Nose2Neck = 0
    Head2Nose = 1
    Shoulder2Shoulder = 2
    Hip2Hip = 3
    Ear2Nose = 4
    Shoulder2Neck = 5
    Elbow2Shoulder = 6
    Wrist2Elbow = 7
    Hip2Shoulder = 8
    Knee2Hip = 9
    Ankle2Knee = 10
    Toe2Ankle = 11
    Heel2Ankle = 12

    # Bones for optional keypoints
    RacketHead2Tail = 13

    NumBones = 13
    NumBonesWithRacket = 14

    @staticmethod
    def expand_stats(arr):
        idxs = [
            Bone_Symmetric.Nose2Neck,
            Bone_Symmetric.Head2Nose,
            Bone_Symmetric.Shoulder2Shoulder,
            Bone_Symmetric.Hip2Hip,
            Bone_Symmetric.Ear2Nose,
            Bone_Symmetric.Ear2Nose,
            Bone_Symmetric.Shoulder2Neck,
            Bone_Symmetric.Shoulder2Neck,
            Bone_Symmetric.Elbow2Shoulder,
            Bone_Symmetric.Elbow2Shoulder,
            Bone_Symmetric.Wrist2Elbow,
            Bone_Symmetric.Wrist2Elbow,
            Bone_Symmetric.Hip2Shoulder,
            Bone_Symmetric.Hip2Shoulder,
            Bone_Symmetric.Knee2Hip,
            Bone_Symmetric.Knee2Hip,
            Bone_Symmetric.Ankle2Knee,
            Bone_Symmetric.Ankle2Knee,
            Bone_Symmetric.Toe2Ankle,
            Bone_Symmetric.Toe2Ankle,
            Bone_Symmetric.Heel2Ankle,
            Bone_Symmetric.Heel2Ankle,
        ]
        if arr.shape[-1] == Bone_Symmetric.NumBonesWithRacket:
            idxs.append(Bone_Symmetric.RacketHead2Tail)
        return arr[idxs]