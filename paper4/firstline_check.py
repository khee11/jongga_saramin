from utils import *
import time

def firstline_checker(data_type):

    info = info_return(data_type)
    if info["skip_first"]:
        print("This dataset ({}) originally does not include column header as its 1st line".format(data_type))
        time.sleep(1)
    
    print("\nFor train.csv, 1st line is :")
    with open(info["path"]+"/train.csv","r") as f:
        for line in f:
            print(line)             
            break

    print("\nFor test.csv, 1st line is :")
    with open(info["path"]+"/test.csv","r") as f2:
        for line in f2:
            print(line)
            break
            
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str,choices=["dbpedia","agnews","yahoo","yelp"])
    args = parser.parse_args()
    
    firstline_checker(args.data)
    
