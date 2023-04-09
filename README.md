---
title: Sentiment Analysis
emoji: 😻
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
license: unknown
---

# csuy-4613-Project
Milestone 1

The operating system that is being used is Windows 10 Home. In order to run Docker on this operating system, a
Windows Subsystem for Linux (WSL) must be used.

WSL was installed by using the wsl --install command in the Windows Command Prompt. This installed the Ubuntu 
distribution of Linux. To set the WSL version to WSL 2, the comman wsl --set-version Ubuntu 2 was used.

The Docker app was installed from the docker website. The option for WSL 2 had to be verified in the
Docker setting. Running wsl.exe -l -v checks the versions of the distro, which we could verify that
the distro Ubuntu ran in version 2. 

We set Ubuntu as the default distro with the command wsl --set-default ubuntu

Using VSCode as the coding environment, we can enter WSL by using wsl and code in the terminal. From there
a linux command prompt can be seen, using ~ to accept new commands. Running docker run hello-world verifies
that the docker is working.

![docker](https://user-images.githubusercontent.com/33811542/227808275-baf0dec3-181c-4b04-beeb-b42c35667edb.jpg)

Milestone 2
