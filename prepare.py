"""Build registered representations and indexes for selected real datasets."""
import argparse
from backend.registry import read_config
from backend.pipeline import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--indexes',nargs='+')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    config=read_config(args.config)
    for field in ('datasets','methods','indexes'):
        requested=getattr(args,field)
        if requested and set(requested)-{e['id'] for e in config[field]}:
            parser.error(f'Unknown {field} selection')
    prepare(config,args.datasets,args.methods,args.indexes,args.force)


if __name__=='__main__': main()
