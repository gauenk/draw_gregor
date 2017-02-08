#!/bin/bash
for i in {1..100}; do th main.lua && break || sleep 5; done
