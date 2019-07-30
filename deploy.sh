#!/bin/bash
echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Go To .git root directory
cd ~/workspace/dnn_hsic

# Add all changes to git.
git add .

# Commit changes.
msg="update repo `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master
