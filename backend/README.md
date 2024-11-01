---
title: Backend
emoji: 🥇
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 5.1.0
app_file: app.py
pinned: true
license: apache-2.0
---

Depending on whether you want to use lighteval or lm_eval for your evaluations, you might need to complete the 
requirements.txt file to contain relevant dependencies.

You'll also need to select, in app.py, whether you want to use the ligtheval or lm_eval by selecting the correct 
import and commenting the other.

All env variables that you should need to edit to launch the evaluations should be in `envs`.