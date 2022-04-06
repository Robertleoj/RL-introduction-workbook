
# Exercise 3.1

> Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as different from each other as possible. The framework is abstract and flexible and can be applied in many different ways. Stretch its limits in some way in at least one of your examples.  

## Controlling a rocket
A reinforcement learning agent is to control a rocket that should fly into orbit. 

The agent will want to know all about the forces acting on the rocket, and the current position, velocity, acceleration, heading, and rotation forces of the rocket. It might need to know the fuel level, and the temperature of the engine in order to complete its task.

Finally, most rockets have discrete stages; the first transition might perhaps be to discard the level 1 boosters. 

Thus the state might be composed of
```
Position
Velocity
Acceleration
Fuel status
Engine temperature
Heading
Rotation
The phase of the launch
```

The agent should control the engine and controls of the rocket. Maybe the agent also controls when the rocket releases its boosters. The agent might also want to warn the humans that something is wrong, so it might want to send a distress signal.

Thus the actions could be
```
Accelerate engine
Turn (or something to control the heading of the engine)
Transition into next phase of launch
Send distress signal
```

The agent should fly the rocket so that it heads forward, with minimum turbulance, with fuel efficiency, and a nice ascent into orbit. Its reward might be some weighted combination of these factors. 



## Programming

We could face a reinforcement learner create programs for tasks. In fact, this has already been done. 

The agent might receive as a state the programming language it should use, if it should not decide that itself. Of course it should want a description of the program. 

An important thing to know when programming is the current state of the code, so the agent should receive that as well. 

Now, the agent might want to run the program to see if it produces the desired output. So that might be a possible part of the state. 

For the actions, the agent might choose to input a character to the file, delete text, or any other aciton that modifies the text. The agent might also decide to run to the program to see the results. Finally, when the agent is happy, it should be able to submit its code. 


## Cook food
We might produce a chef agent. It might control some kind of robot that has access to all necessary appliances in a kitchen, and the ingredient, and the agent's goal is to produce the requested dish. 

The states might consist of visual input, temperature of stoves and ovents, and the sensorymotor data from the robot's actuators. 

The agent needs to control the robot, so the actions should include all the possible controls of the robot. Also, for better automation, stove and oven controls might be actions seperate from the robot's movement. 

The rewards should be negative for breaking stuff, or otherwise reaking havoc in the kitchen, and there should be positive rewards for creating the requested dish, where the size of the reward is a funciton of the dish's quality.

# Exercise 3.2
> Is the MDP framework adequate to usefully represent all goal-directed learning tasks? Can you think of any clear exceptions?

I cannot find any counterexamples.


# Exercise 3.3

The actions that the agent can perform instantaneously at any time step should be the defined actions for the agent. The choice of where to drive should not be an output for a driving agent, as that is not how you drive a car, in the immediate sense. 

The choice of where to drive should be something that the agent decides makes plans on how to execute, and then uses its actuators/available actions to perform. 

A driving agent that would output a coordinate of where to drive would not be a driving agent, rather a guidance agent. 

Of course, a driving agent could have some other agent that controls where it should drive, and the driving agent implements that plan, but that would be a seperate agent, distinct form the driving agent. 

Now, as stated, the actions should be something that can be outputted, and executed immediately. If the agent outputs a rotational acceleration of the steering wheel, or simply its goal angle should be something that could be immediately implemented by some microcontroller or other machinery. 

Therefore that is an acceptable action. 

Now, if there is some automatic system that can better automate a task than the agent can, say, a good brake system in a car that prevents skidding, the agent should not directly control the force of the brake pads, but should simply gain access to the brake pedal, and let the brake system do the rest.

# Exercise 3.4
   
  
| s      |  a        |  s'   | r     | $p(s', r \mid s, a)$  |    
| ----   | -----     | ----  | ----- | -----------------     |    
|  high  | search    |  high |  1    |   0.4                 |  
|  high  | search    |  high |  0    |   0.6                 |  
|  high  | search    |  low  |  1    |   0.4                 |  
|  high  | search    |  low  |  0    |   0.6                 |  
|  high  | wait      |  high |  1    |   0.1                 |  
|  high  | wait      |  high |  0    |   0.9                 |  
|  high  | wait      |  low  |  1    |   0.1                 |  
|  high  | wait      |  low  |  0    |   0.9                 |  
|  low   | search    |  low  |  1    |   0.2                 |  
|  low   | search    |  low  |  0    |   0.8                 |  
|  low   | wait      |  low  |  1    |   0                   |  
|  low   | wait      |  low  |  0    |   1                   |  
|  low   | recharge  |  low  |  1    |   0                   |  
|  low   | recharge  |  low  |  0    |   1                   |  


# Exercise 3.5
The formula simply becomes 

$$
p(s', r \mid s, a) = 1, \text{ for all } s \in \mathcal{S}\setminus \mathcal{S}^+
$$
   
# Exercise 3.6
The return would not just be related to $-\gamma^K$, it would equal $-\gamma^K$, where $K$ is the number of steps until failure.

In the continuing task formulation, the agent might also take into account the next failure, in the next reset, in a given state. 


# Exercise 3.7
The expected return is always one, as the agent always escapes the maze eventually, and when it does, there is no discrimination between states. All states get an expected return of one. 

To communicate to the agent that it should navigate the maze efficiently, we should penalize it for every move it makes, so that it wants to escape the maze quickly. We only instructed it to *eventually* escape the maze, no matter how long it takes. 

# Exercise 3.8
$$
G_5 = 0\\
G_4 = R_5 + \gamma G_5 = 2 + 0.5 \cdot 0 = 2\\
G_3 = R_4 + \gamma G_4 = 3 + 0.5 \cdot 2 = 4\\
G_2 = R_3 + \gamma G_3 = 6 + 0.5 \cdot 4 = 8\\
G_1 = R_2 + \gamma G_2 = 2 + 0.5 \cdot 8 = 6\\
G_0 = R_1 + \gamma G_1 = -1 + 0.5 \cdot 6 = 2
$$

# Exercise 3.9
We know that
$$
G_1 = \sum_{i=0}^{\infty} 7\gamma^i = 7\sum_{i = 0}^{\infty}\gamma^i\\
= 7 \frac{1}{1 - \gamma} = 7\frac{1}{0.1} = 7 \cdot 10 = 70
$$
And 
$$
G_0 = 2 + \gamma G_1 = 2 + 0.9 \cdot 70 = 2 + 7 = 65
$$

# Exercise 3.10
Let 
$$ 
\Gamma = \sum_{k = 0}^{\infty}\gamma^k
$$
Then 
$$
\Gamma - \gamma \Gamma = \sum_{k = 0}^{\infty}\gamma^k -  \sum_{k = 1}^{\infty}\gamma^k = \gamma^0 = 1
$$
And hence
$$
(1 - \gamma) \Gamma = 1\\
\implies \Gamma = \frac{1}{1 - \gamma}
$$







