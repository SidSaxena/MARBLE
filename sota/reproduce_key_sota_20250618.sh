rm -rf output/probe.MergeGSHT.MuQ
python cli.py fit -c sota/key_sota_20250618.yaml
python cli.py test -c sota/key_sota_20250618.yaml