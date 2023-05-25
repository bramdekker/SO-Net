import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_som(file):
    data = np.load(file)

    som_node_np = data['som_node'] # 3 x 64 (node_num) (8x8 or 4x4)
    pc_np = data['pc']

    print(f"Shape of som_node_np is {som_node_np.shape}")
    print(f"Shape of pc_np is {pc_np.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(som_node_np, c='r')
    ax.scatter(pc_np, c='b')

    plt.show()


if __name__  == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('--dir', '-d', help="Path to directory containing LAS files", type=str)
        parser.add_argument('--file', '-f', help="Path to a LAS file", type=str)

        args = parser.parse_args()

        if args.dir == None and args.file == None:
            print("Specify either directory or file")
        elif args.dir != None and args.file != None:
            print("Specify only one of file and dir.")
        elif args.dir != None:
            for f in os.listdir(args.dir):
                visualize_som("%s/%s" % (args.dir, f))
        elif args.file != None:
            visualize_som(args.file)
        
        