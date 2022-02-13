# Telegram Bot using Fast Neural Style Transfer
This repository contains Telegram Bot implementation, that uses for Style Transfer. Style Transfer implementation entirely based on Pytorch Fast Neural Style implementation, shown below. 

- [PyTorch Fast Neural Style Example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

This repo contains other styles, besides the standart ones, contains in Pytorch Fast Neural Style repository. The style transfer models will be gradually added, as will be trained.

For now available styles is: 
- colormix 1
- lights 1, 2, 3
- psychodelic 3
- all standart torch styles

## Requirements
- torch
- torchvision
- pyTelegramBotAPI
- pillow

### Quality
For the purpose of make possible deploy this app on heroku, the image size of pictures will be decreased to 512 px in width. If it'll be a purpose to get better results, you can clone git and perform stylize on your own.

### Config
Config file was not added as it contains API key for TelegramBot
