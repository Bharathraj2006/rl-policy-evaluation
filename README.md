# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
To find best policy from two policies which are defined by user using policy evaluation function. Where the mdp includes 16 states from 0-15, 0 is the starting state, assigning some 4 random state as holes and 15 is the goal state and then we need to calculate optimal state value function for each state such that we can reach goal using optimal policy using policy evaluation.

## POLICY EVALUATION FUNCTION
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
Mention the first and second  along with its state value function and compare them
### policies:
![alt text](output/policy.png)

### State value function:
![alt text](output/state1.png)
![alt text](output/state2.png)

### Compare:
![alt text](output/compare.png)

### Best Policy:
![alt text](output/finalout.png)

## RESULT:
Thus, The Python program to evaluate the given policy is successfully executed.
