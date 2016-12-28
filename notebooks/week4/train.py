from model import DCGAN
from solver import Solver

def main():
    model = DCGAN()
    solver = Solver(model, num_epoch=10, image_path='data/celeb_resized')
    solver.train()
    

if __name__ == "__main__":
    main()