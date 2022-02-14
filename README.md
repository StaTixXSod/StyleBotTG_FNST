# Telegram Bot using Fast Neural Style Transfer
This repository contains Telegram Bot implementation, that uses for Style Transfer. Style Transfer implementation entirely based on Pytorch Fast Neural Style implementation, shown below. 

- [PyTorch Fast Neural Style Example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

## Bot
You can find this Telegram Bot after searching for "@StatixX_StyleBot".
This repo contains other styles, besides the standart ones, contains in Pytorch Fast Neural Style repository. The style transfer models will be gradually added, as will be trained.

## Learning
TransformerNet was taken from Pytorch Fast Neural Style github repo and was trained on COCO dataset with 1 epoch. It seems like it's reallyy small amount of training process... BUT! On my GTX 1050 4GB I was able to train models only with 2 image batch size and 1 epoch was lasted around 3 hours with 256 px of image size. 
But anyway, may be some day i'll get rid of "bad" styles and train "good" styles a bit longer.
Actually I've noticed, that style pictures with eye-catching edges performs better on images, than style pictures with faded background. But you can still try all of available styles.

## Training process
There is important thing to do, all pictures in repo sorted and named as "styles/style/style#number.jpg". And the same thing with models: "models/style/style#number.model". Because of this kind of sorting "train_styles.py" finds all styles in folder "styles" and after training model on each style "train_styles.py" saves models to corresponding folders. 
So if you have your style pictures, you can add a folder with new pictures and run training process.

## Available styles 
- colormix 1, 2, 3
- lights 1, 2, 3
- psychodelic 3
- all standart torch styles

## Advice
After training a bunch of styles I can say, you should pick styles with expressed edges. Iridescent colors not really good for styling. And one more thing... my favourite style is Psychodelic 3. Try it out! 

## Requirements
- torch
- torchvision
- pyTelegramBotAPI
- pillow

### Quality
For the purpose of make possible deploy this app on heroku, the image size of pictures will be decreased to 512 px in width. If it'll be a purpose to get better results, you can clone git and perform stylize on your own.

### Config
Config file was not added as it contains API key for TelegramBot
