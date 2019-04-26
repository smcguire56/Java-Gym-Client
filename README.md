# B. Sc. Software Development –Applied Project & Minor Dissertation 
### Applying Reinforcement Learning techniques to the Cart-Pole problem using Q-Learning

## Introduction

This is a project for GMIT's Applied Project & Minor Dissertation module. This  research  project  is  based  on  the  Q-learning,  and  how this  can  be  applied  to  reinforcement  learning  environments.   This  concept was  applied  to  an  environment  provided  by  the  gym  library  developed  byOpen AI, in order to demonstrate its potential.  This is developed using Java as the client and the gym HTTP API as the server which displays the Cart-Pole environment.

## Getting Started

To get this project to work locally first download this repository:

```
https://github.com/smcguire56/Java-Gym-Client
```

Then clone down the Python server: 
```
git clone https://github.com/openai/gym-http-api
```

To run the Python server make sure you run these commands to install the neccessary libraries:

```
cd gym-http-api
pip install -r requirements.txt
```

After these libraries have been installed you may run the Python server locally using the "gym_http_server.py" file:

```
python gym_http_server.py
```

A pop up window should display: "Server starting at: http://127.0.0.1:5000", that means the server is running correctly.
After that open up thr Eclipse IDE and import the "Java-Gym-Client" code downloaded earlier. Once we run this Java code we can see the Client interacting with the Server and a Python window should pop up of the CartPole. 


## Authors

* **Sean McGuire** - [GitHub](https://github.com/smcguire56)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


