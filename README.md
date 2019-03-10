## Welcome to ProjectFin

This repository contains some scripts and interfaces for finance

`mysite` is a Web User interface written in Django that allows to visualize stock prices and crypto prices times series, predict future directions with machine learning algorithms, and implement simulations such as monte carlo.

`crypto-recorder` contains several scripts and notebooks for cryptocurrencies analysis

`quickfix-tradeclient-django` and `quickfix-tradeclient-golang` are web user interfaces 
These were forked from https://github.com/SSOC18, where the whole quickfix messaging protocol was build.

With the quickfix-executor and rabbitmq message queuer, the web user interfaces can be used to send FIX messages.
A new order transaction example in YouTube: https://youtu.be/K6vgXpXaFd0.
It shows the executor accepting and filling a new order from the Web UI tradeclient through the RabbitMQ server.
