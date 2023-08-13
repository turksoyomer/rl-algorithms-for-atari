from utils.trainer import Trainer

env_id = "ALE/Breakout-v5"
algorithm = "PPO"
num_frames = 20000
last_trained_frame = 20000
saving_interval = 10000

trainer = Trainer(env_id, algorithm, num_frames, last_trained_frame, saving_interval)

trainer.plot(last_trained_frame)
