import click
import multiprocessing

from src.generate_data import generate_data
from src.generate_plots import generate_plots


@click.command()
# experiment modes
@click.option('--gradient', is_flag=True, help='Flag for repeated gradient ascent method')
@click.option('--sampling', is_flag=True, help='Flag for finite samples')
# experiment parameters
@click.option('--eps', default=.3, type=float, help='Environment parameter epsilon')
@click.option('--fbeta', default=5, type=float, help='Fixed value for (smoothness) parameter beta')
@click.option('--betas', multiple=True, default=[.1, 1, 5, 10, 200], type=float, help='List of values for (smoothness) parameter beta')
@click.option('--flamda', default=1, type=float, help='Fixed value for (regularization) parameter lambda')
@click.option('--lamdas', multiple=True, default=[0, .2, .5, 1, 5], type=float, help='List of values for (regularization) parameter lambda')
@click.option('--fgamma', default=.9, type=float, help='Fixed value for discount factor gamma')
@click.option('--gammas', multiple=True, default=[], type=float, help='List of values for discount factor gamma')
@click.option('--freg', default='L2', type=click.Choice(['L2', 'ER']), help='Fixed value for regularizer')
@click.option('--regs', multiple=True, default=[], type=click.Choice(['L2', 'ER']), help='List of values for regularizer')
@click.option('--num_followers', default=50, type=int, help='Number of followers')
# gradient
@click.option('--feta', default=1, type=float, help='Fixed value for (step size) parameter eta')
@click.option('--etas', multiple=True, default=[.05, .1, .2, 1, 2], type=float, help='List of values for (step size) parameter eta')
# sampling
@click.option('--fn_sample', default=200, type=int, help='Fixed value for number of samples')
@click.option('--n_samples', multiple=True, default=[20, 50, 100, 200, 500, 1000], type=int, help='List of values for number of samples')
@click.option('--num_seeds', default=20, type=int, help='Number of experiment seeds')
# iterations
@click.option('--max_iterations', default=1000, type=int, help='Number of Iterations')
# n_jobs
@click.option('--n_jobs', default=multiprocessing.cpu_count(), type=int, help='Number of jobs')
# policy gradient
@click.option('--policy_gradient', is_flag=True, help='Flag for policy gradient method')
@click.option('--nus', multiple=True, default=[.1, .2, 1, 2, 5], type=float, help='List of values for (policy gradient) parameter nu')
# unregularized objective
@click.option('--unregularized_obj', is_flag=True, help='Flag for evaluation on unregularized objective')
# lagrangian
@click.option('--lagrangian', is_flag=True, help='Flag for solving the lagrangian')
@click.option('--n', default=10, type=int, help='Number of rounds N')
@click.option('--delta', default=.1, type=float, help='Lagrangian parameter delta')
@click.option('--b', default=10, type=int, help='Lagrangian parameter B')
def run_experiment(gradient, sampling, eps, fbeta, betas, flamda, lamdas, fgamma, gammas, freg, regs, num_followers, feta, etas, fn_sample, n_samples, num_seeds, max_iterations, n_jobs, policy_gradient, nus, unregularized_obj, lagrangian, n, delta, b):

    print("Begin experiment\n")

    params = {}
    params['gradient'] = gradient
    params['sampling'] = sampling
    params['eps'] = eps
    params['fbeta'] = fbeta
    params['betas'] = betas
    params['flamda'] = flamda
    params['lamdas'] = lamdas
    params['fgamma'] = fgamma
    params['gammas'] = gammas
    params['freg'] = freg
    params['regs'] = regs
    params['num_followers'] = num_followers
    params['feta'] = feta
    params['etas'] = etas
    params['fn_sample'] = fn_sample
    params['n_samples'] = n_samples
    params['seeds'] = list(range(num_seeds))
    params['max_iterations'] = max_iterations
    params['n_jobs'] = n_jobs
    # policy gradient
    params['policy_gradient'] = policy_gradient
    params['nus'] = nus
    # unregularized objective
    params['unregularized_obj'] = unregularized_obj
    # lagrangian
    params['lagrangian'] = lagrangian
    params['N'] = n
    params['delta'] = delta
    params['B'] = b

    if lagrangian:
        assert sampling, "Lagragian is solved for the finite sample case!"
    assert (gradient or (sampling and not lagrangian)) + policy_gradient + unregularized_obj + lagrangian <=1, "Only use one solution concept is allowed (see README file)!"

    generate_data(params)
    generate_plots(params)

    print("Finish experiment")

if __name__ == '__main__':
    
    run_experiment()