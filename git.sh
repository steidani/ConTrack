#!/bin/bash  
git add -A  
read -p "Commit description: " desc  
git commit -m "$desc"
git push -u origin master
