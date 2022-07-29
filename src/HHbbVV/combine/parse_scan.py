"""
Scrapes through the scan folders to find the significances.

Author: Raghav Kansal
"""

from os import listdir

cut_dirs = [f for f in listdir("./") if f.is_dir()]
