Deep RL Assignment 1: Imitation Learning

## Section 1. Getting Set Up

Note that OpenAI gym should be version 0.9.1 to be compatible.



## Section 2. Warmup

Run:

```
# Generate expert rollout data
python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20 --save_data

# Train Behavioral Cloning agent (generates graph below)
python train_bc.py experts/Hopper-v1-data.pkl Hopper-v1

# Run Behavioral Cloning agent
python run_bc.py experts/Hopper-v1.pkl Hopper-v1 --ckpt_dir bc_ckpt --render
```

<img src="bc_ckpt/Hopper-v1.png" width="400">

Figure. Loss vs Time steps (1K) of Behavioural Cloning on Hopper-v1.



## Section 3 & 4. Behavioral Cloning & Dagger

Example run:

```
# Generate expert rollout data
python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20 --save_data

# Train & run Behavioral Cloning agent
python train_bc.py experts/Ant-v1-data.pkl Ant-v1
python run_bc.py experts/Ant-v1.pkl Ant-v1 --ckpt_dir bc_ckpt --render

# Train & run DAgger agent
python train_dagger.py experts/Ant-v1.pkl experts/Ant-v1-data.pkl Ant-v1
python run_bc.py experts/Ant-v1.pkl Ant-v1 --ckpt_dir dag_ckpt --render
```

| Task           | Behavioral Cloning | Dagger            | Expert         |
| -------------- | ------------------ | ----------------- | -------------- |
| Ant-v1         | **852.4** ± 12.7   | 630.9 ± 64.1      | 4755.7 ± 476.1 |
| Reacher-v1     | -12.6 ± 4.1        | **-10.0** ± 4.9   | -4.2 ± 1.5     |
| Walker2d-v1    | 71.1 ± 16.8        | **294.3** ± 87.2  | 5519.9 ± 66.4  |
| Hopper-v1      | **67.1** ± 1.4     | 63.7 ± 2.7        | 3778.9 ± 3.2   |
| HalfCheetah-v1 | 92.2 ± 82.4        | **515.3** ± 578.7 | 4152.3 ± 90.2  |

Table. Mean and Standard Deviation (over 20 rollouts) of rewards on different tasks after training.

