from args import args
import pblks, cherry, cl4phi

if __name__ == '__main__':

    if args.model == 'pblks':
        runner = pblks.PBLKSRunner(args)
    elif args.model == 'cl4phi':
        runner = cl4phi.CL4PHIRunner(args)
    elif args.model == 'cherry':
        runner = cherry.CHERRYRunner(args)
    runner.run()
