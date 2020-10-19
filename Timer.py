# Timer Class
import time

class Timer:
    '''class for timing operations'''

    def __init__(self, active=True, num=None, name=None, readout=False, roundto=None):
        '''initializes the timer class with an activation status (bool),
        a number, or a name'''
        self.t0 = time.time() # record the time when the class was initizlized

        # make sure all arguments are correct data types
        if active and type(active) != bool:
            raise AttributeError('Variable active must be type bool, not '+str(type(active)))
        if num and type(num) != int:
            raise AttributeError('Variable num must be type int, not '+str(type(num)))
        if name and type(name) != str:
            raise AttributeError('Variable name must be type string, not '+str(type(name)))
        if readout and type(readout) != bool:
            raise AttributeError('Variable readout must be type bool, not '+str(type(readout)))
        if roundto and type(roundto) != bool:
            raise AttributeError('Variable roundto must be type int, not '+str(type(roundto)))

        # figure out how the timer should be introduced when printing to the console
        if num and name:
            self.intro = str(name)+' (#'+str(num)+'): '
        elif num and not name:
            self.intro = 'Timer #'+str(num)+': '
        elif name and not num:
            self.intro = str(name)+': '
        else:
            self.intro = 'Timer: '

        # figure out how many digits to round to
        if roundto:
            self.roundto = roundto
        else:
            self.roundto = 3

        # initialize some other variables
        self.label = None # current label (initialized to None)
        self.active = active # activation status
        self.record = [('init',)] # record of all readouts
        self.intervaln = 0 # interval number
        self.readoutn = 0 # readout number

        # initialize this dict for printing the log
        self.printfuncs = {
            'init': self.__print_init__,
            'interval': self.__print_interval__,
            'readout': self.__print_readout__
        }

        if readout: # readout initialization, if activated
            print(self.intro+'Initialized at t = '+str(round(self.t0,self.roundto)))

    def turnon(self):
        '''turns the timer on'''
        self.active = True

    def turnoff(self):
        '''turns the timer off'''
        self.active = False

    def readout(self, label=None):
        '''prints out a label with the elapsed time since t0'''
        t = time.time() # current time
        elapsed = round(t - self.t0, self.roundto) # time since starting
        if label: # if theres no label, just give it a number
            label = str(label)
        else: # otherwise keep the given label
            label = '#'+str(self.readoutn)
        self.record.append(('readout', self.readoutn, label, elapsed)) # log to record
        self.readoutn += 1 # add one to the readout number
        if self.active: # if active, readout the thing
            print(self.intro+label+', time = '+str(elapsed)+' sec')

    def start(self, label=None):
        '''start the timer with a certain label'''
        if label:
            self.label = str(label) # make label a string
        else:
            self.label = 'readout #'+str(self.intervaln)
        self.tstart = time.time()-self.t0 # get the time of starting
        if self.active: # if active, readout
            print(self.intro+self.label+'...')

    def end(self):
        '''end the timer on the current task'''
        self.tend = time.time()-self.t0 # record ending time ASAP
        if self.label == None: # if timer hasn't been started, throw an error
            raise AttributeError('Timer has not been started')
            return None
        elapsed = round(self.tend-self.tstart, self.roundto) # time elapsed since starting
        self.record.append(('interval', self.intervaln, self.label, self.tstart, self.tend, elapsed)) # record all to the log
        if self.active: # readout, if active
            print(self.intro+'Done '+self.label+', took '+str(elapsed)+'sec')
        self.label = None # reset label to none
        self.intervaln += 1 # add one to the interval number
        pass

    def __print_init__(self, tuple):
        '''prints the initialization time'''
        print('Initialized at t = ',str(round(self.t0, self.roundto)))

    def __print_interval__(self, tuple):
        '''prints the record of an interval to the log'''
        _,n,lbl,tst,tnd,eps = tuple
        print('INTERVAL #'+str(n)+': label='+lbl+', tstart='+str(tst)+', tend='+str(tnd)+', telapsed='+str(eps))

    def __print_readout__(self, tuple):
        '''prints the record of a readout'''
        _,n,lbl,t = tuple
        print('READOUT #'+str(n)+': label='+lbl+', t='+str(t))

    def printlog(self):
        '''prints the entire record of the timer to the console'''
        print('\n' + self.intro + ' PRINTING LOG')
        for tuple in self.record:
            self.printfuncs[tuple[0]](tuple)
        print(self.intro + ' END OF LOG' + '\n')










#
