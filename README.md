# POLICY EVALUATION

## AIM
To simulate the Frozen-lake MDP and compare different policy functions.

## PROBLEM STATEMENT
The problem involves simulating a Frozen-lake MDP and defining various policy functions for it, these policy functions are later evaluated by a policy_evaluation() function which compares the value function of the policies passed as parameter. This is an experiment in reinforcement learning where you test different policies in FrozenLake, both by simulation (probability of reaching the goal) and by formal policy evaluation (computing expected long-term rewards).

## POLICY EVALUATION FUNCTION

<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/834db01d-47b9-40d8-895e-5b7fc488ed1d" />

```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:

### policies:
![alt text](output/image.png)
![alt text](<output/image copy.png>)
![alt text](<output/image copy 2.png>)
![alt text](<output/image copy 3.png>)

### State value function:
![alt text](<output/image copy 4.png>)

### Compare:
![alt text](<output/image copy 5.png>)

### Best Policy:
![alt text](<output/image copy 6.png>)

## RESULT:
Thus we have successfully evaluated two different policies for a given env and compared their values functions.