
![Frame 1(2)](https://github.com/user-attachments/assets/8ad9f278-dc94-4e21-be03-4fac26b738db)

# LIGHT CONTROLL

Light controll is a **node based** lighting (SACn) controll software

## Motivation

Due to the lack of modern open source lighting software or extensibility of industry standard software, we decided to create a modern node based lighting software.  
The goal is to create a platform that:
- is easy to use
- is easy to extend with custom features
- comes "batteries included" with basic features

## Alternatives

Here are some open source alternatives:

- [QLC+](https://www.qlcplus.org/)
- [OpenRGB](https://openrgb.org/)
- [DMXcontroll](https://www.dmxcontrol.de/de/)
- [Glight](https://github.com/aroffringa/glight)

## Light controll UI

![Graph UI](https://github.com/user-attachments/assets/9385f09c-e6f1-4416-b2c3-46b4356d035a)  
![Node search](https://github.com/user-attachments/assets/b85110e8-9795-400c-8595-0f5f371bcd87)  
![image](https://github.com/user-attachments/assets/7292ff60-7cf7-478f-9ad2-23ae2dba6086)  



## How to run
### Using python

- Clone project using `git clone https://github.com/gerbiceva/light_control.git`
- cd into project `cd light_control/server`
- run the script using [UV](https://astral.sh/blog/uv) `uv run server.py`

you should see an output like:  
```py
INFO:     Started server process [21926]
INFO:     Waiting for application startup.
INFO:     Started server process [21926]
INFO:     Waiting for application startup.
INFO:     ASGI 'lifespan' protocol appears unsupported.
INFO:     Application startup complete.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:50051 (Press CTRL+C to quit)
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```
Application frontend is now available at [http://127.0.0.1:8080](http://127.0.0.1:8080)

## How to develop
### GRPC  client / server generation

- install requirements for `python` and `typescript`
  - inside `frontend` run `yarn` to install dependencies
  - inside `src` either make a `venv` or use global python install. To install dependencies run `pip install -r requirements.txt`
- inside `gRPC` folder there are scripts to recompute server and client grpc code.
  - to install or update everything, `cd` into it and run `update_grpc.bash`
