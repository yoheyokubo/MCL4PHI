from args import args
import pblks, cherry, cl4phi, gumbel

if __name__ == '__main__':

    if args.model == 'pblks':
        runner = pblks.PBLKSRunner(args)
    elif args.model == 'cl4phi':
        runner = cl4phi.CL4PHIRunner(args)
    elif args.model == 'cherry':
        runner = cherry.CHERRYRunner(args)
    elif args.model == 'gumbel':
        runner = gumbel.GumbelRunner(args)
    #elif args.model == 'phagetb':
    #    runner = phagetb.PHAGETBRunner(args)
    #elif args.model == 'graph':
    #    runner = graph.GRAPHRunner(args, model)
    runner.run()
