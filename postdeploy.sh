#!/bin/bash

# Install git-lfs if not already
apt-get update && apt-get install -y git-lfs

# Initialize and pull LFS files
git lfs install
git lfs pull
