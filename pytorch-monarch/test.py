from monarch.actor import Actor, endpoint, this_host

# spawn 2 trainer processes one for each cpu
training_procs = this_host().spawn_procs({"cpus" : 2})

# define the actor to run on each process 
class Trainer(Actor):
    @endpoint 
    def train(self, step: int): ...


# create the trainers 
trainers = training_procs.spawn("trainers", Trainer)

# tell all the trainers to take a step 
fut = trainers.train.call(step=0)

# wait for all the trainers to finish 
fut.get()