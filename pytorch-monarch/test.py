from monarch.actor import Actor, endpoint, this_host

# spawn 2 trainer processes one for each cpu
procs = this_host().spawn_procs({"cpus" : 32})

print("-" * 40)
for key, val in procs.__dict__.items():
    print(f"{key} ---> {val}")
print("-" * 40)

# define the actor that has two methods

class Example(Actor):
    @endpoint 
    def say_hello(self, txt):
        return f"hello {txt}"
    
    @endpoint
    def say_bye(self, txt):
        raise Exception("saying bye is hard")


# spawn the actors 
actors = procs.spawn("actors", Example)

"""
# have half of them say hello 
hello_fut = actors.slice(cpus=slice(0, 3)).say_hello.call("world")

# have half of them say goodbye
bye_fut = actors.slice(cpus=slice(3, 4)).say_bye.call("world")

try:
    print(hello_fut.get())
except:
    print("couldn't say hello")
    
try:
    print(bye_fut.get())
except Exception:
    print("got an exception saying bye")
"""