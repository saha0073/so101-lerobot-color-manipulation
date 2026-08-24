policy.n_action_steps = making to 25 from 50 does not improve the jittering, instead the pick & place seems to be working better at 50
At 25, e.g. the gripper is holding the object to move from point A to B and in-between when the 25 steps ends it tries to predict and then failing the object

policy.num_steps = 20  -- making it 20 from 10 made decent improvement in the jittering