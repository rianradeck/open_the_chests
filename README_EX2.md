i used has reference:  https://arxiv.org/pdf/2106.01345

lib used: pytorch -->tried to do embedding of all the stack features of an event

helpers created: sequence generator to train the transformer from scratch

plots: test a random sequence 

training:  
    python -m open_the_chests.cli.dt_train --env << >> --num-sequences 500 --n-events 200 --epochs 250
    GPU:
      0  NVIDIA GeForce RTX 5060 ...    On  |   00000000:01:00.0  On |                  N/A |
    | N/A   71C    P2             84W /  103W |    2439MiB /   8151MiB |     92%      Default |
    
evaluation:

    python -m open_the_chests.cli.dt_eval evaluate  --n-events 200 --env easy
{'loss': 0.000130862371796476, 'accuracy': 0.9999999999999167, 'precision': 0.9999999999833333, 'recall': 0.9999999999833333, 'f1': 0.9999999949833334}

    python -m open_the_chests.cli.dt_eval evaluate  --n-events 200 --env medium
{'loss': 0.00011046543139465419, 'accuracy': 0.9999999999999167, 'precision': 0.9999999999833333, 'recall': 0.9999999999833333, 'f1': 0.9999999949833334}

    python -m open_the_chests.cli.dt_eval evaluate  --n-events 200 --env hard
{'loss': 0.002214645443018526, 'accuracy': 0.99943333333325, 'precision': 0.9067278287323131, 'recall': 0.9883333333168611, 'f1': 0.945773519715081}


tests: 
    graphs