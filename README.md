# Telegram Bot using Fast Neural Style Transfer

This repository contains Telegram Bot implementation, that uses for Style Transfer. Style Transfer implementation entirely based on Pytorch Fast Neural Style implementation, shown below.

- [PyTorch Fast Neural Style Example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

## Bot

You can find this Telegram Bot after searching for `@StatixX_StyleBot`.
This repo contains other styles, besides the standart ones, contains in Pytorch Fast Neural Style repository, that has been moved into `artwork` directory.

## Requirements

- torch
- torchvision
- pyTelegramBotAPI
- pillow

## Available styles

- psychodelic
- abstraction
- artwork
- photo style

> Available style photos store in the `style` folder, so you can find them there or use the bot, which will show you all styles it has.

## Usage

- Click start
- Upload an image when the bot asks you
- Choose the style you want
- Wait for the stylized image
- Click "Finish" if you're done.

## Learning process

TransformerNet was taken from Pytorch Fast Neural Style github repo and was trained on COCO dataset with 2 epoch on (128 x 128) image size. The `train_styles.py` trains models for all styles, contained in the `style` folder iteratively.


## Examples

_Example 1:_

<img src="https://github.com/StaTixXSod/StyleBotTG_FNST/blob/master/examples/content_examples.jpg?raw=true" width="512" height="256">

_Example 2:_

<img src="https://github.com/StaTixXSod/StyleBotTG_FNST/blob/master/examples/galgadot_examples.jpg?raw=true" width="280" height="280">


### Quality

For the purpose of make possible deploy this app on heroku, the image size of pictures will be decreased to 512 px in width. If it'll be a purpose to get better results, you can clone git and perform stylize on your own.

### Config

Config file was not added as it contains API key for TelegramBot
