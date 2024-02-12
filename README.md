# Probabilistic Artificial Intelligence
## Gaussian Process Regression
<img align="right" height="140" src="https://github.com/majkevh/eth-pai-as23/blob/main/img/gaussianprocess.jpg"></img>
For this task, the goal was to predict the concentration of PM2.5 per cubic meter in the air, given the relative 2D coordinates of locations on a map, along with a binary variable indicating whether that point lies in a residential area.
<br/><br/>

## Approximate Bayesian inference via SWAG
<img align="left" height="140" src="https://github.com/majkevh/eth-pai-as23/blob/main/img/gaussianprocess.jpg"></img>
In this regression task where we aimed to predict the PM2.5 levels based on x-y coordinates and the designation of these coordinates as residential areas. We addressed this task by initially undersampling the data to include only the most relevant data points, achieved through K-means clustering. Subsequently, we fitted a Gaussian Process model with a Matérn kernel and white noise to the data.
<br/><br/>

## Hyperparameter tuning with Bayesian optimization
<img align="right" height="140" src="https://github.com/majkevh/eth-pai-as23/blob/main/img/bo.jpg"></img>
<img align="right" height="140" src="https://github.com/majkevh/eth-pai-as23/blob/main/img/bo1.jpg"></img>
In this task we use Bayesian optimization to optimize drug candidate's structural features for absorption and distribution, aiming for high bioavailability (logP) and easy synthesis. We'll adjust parameter $x$ to balance logP and synthetic accessibility, targeting optimal features $x^*$ with high logP and feasible synthesizability.
<br/><br/>


## Off-policy Reinforcement Learning
<img align="left" height="140" src="https://github.com/majkevh/eth-pai-as23/blob/main/img/pendulum.gif"></img>
In this project we implemented an off-policy reinforcement learning algorithm  to control and swing-up an [inverted pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/).
<br/><br/>