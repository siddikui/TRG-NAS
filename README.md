# TRG-NAS
True Rank Guided Neural Architecture Search for End to End Low Complexity Network Discovery

See Searchlog.txt for search results. 

To run the search locally, use Pytorch>=1.13 and run: 

python search.py --save deeper --gpu 0 --cutout --epochs 600 --min_depth 10 --max_depth 20 --min_width 16 --max_width 72 --width_resolution 4 


