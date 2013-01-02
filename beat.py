#!/usr/bin/python

# Given a stream of accelerometer data on a device moving
# rhythmically (i.e. to a beat) compute the frequency at
# which the device is moving, the force of movement, and
# predict the time of the next beat.
#
# For instance, if you are fist pumping, it will compute
# the tempo of the music that you are following (in Hertz)
# and the time of the next beat.
#
# The technique is based on Singular Value Decomposition (to
# convert 3D motion data to a 1D stream) and Fast Fourier
# Transform (to compute the frequency space).
#
# Reads a stream of "x,y,z" accelerometer data coming in
# on standard input. If getting data via a TCP socket, say
# on port 18250 like the Accelerometer Mouse app does, run
# this script by piping data from socat:
#
#    socat TCP-LISTEN:18250,fork - | python beat.py
#
# Inspired by @TJL's tweet.

import sys
import numpy
import numpy.linalg
import scipy, scipy.fftpack
from datetime import datetime

# Configuration: How often should we re-estimate the sample
# frequency (based on how fast data is coming in, as long
# as we can keep up with it!). This is a number of samples.
sample_time_every = 20

# How many samples should we compute the FFT over? A longer
# history means it takes longer to shift to new rhythms, but
# the frequency will be more accurate because it has a larger
# sample size to be computed from.
sample_history_size = 150

# Some ongoing state. The current estimate of the sample
# frequency:
sample_freq = None

# And the current sample number, incremented consecutively:
current_sample_num = 0

# Buffers
history_time = [] # current time every sample_time_every samples
history_acc = [] # last sample_history_size accelerometer values
history_f0 = [0.0 for i in xrange(4)] # computed frequencies
history_phase = [0.0 for i in xrange(5)] # computed phases

# For predicting beats, the last beat predicted so that we don't
# predict another beat again right away.
last_beat = 0

while True:
	# Read from the device.
	line = sys.stdin.readline().strip().split(",")
	
	# Because I'm testing with the Accelerometer Mouse app,
	# filter out control sequences that are not relevant.
	if line[0] in ("jamjamjam", "paused:"): continue
	
	# Get the accelerometer data and put it into a history array.
	vector = [float(f) for f in line[0:3]]
	history_acc.append(vector)
	if len(history_acc) > sample_history_size: history_acc.pop(0)
	
	# Estimate the sample frequency every 10 samples by the
	# number of samples acquired divided by the time elapsed.
	# Make sure that this is executed on the first iteration.
	if (current_sample_num % sample_time_every) == 0:
		history_time.append(datetime.now())
		if len(history_time) > 10: history_time.pop(0)
		ts = (history_time[-1]-history_time[0]).total_seconds()
		if ts == 0.0: continue
		sample_freq = sample_time_every * float(len(history_time)) / ts
	current_sample_num += 1
	if sample_freq == 0.0: continue # not yet computed

	# There must be something like a 3D FFT, but to use the
	# normal FFT we must compute a single value, not a vector,
	# at each time in the history. We need a method to convert
	# the 3D history data into 1D data.
	
	# Let's assume that the device is moving back and forth
	# along a line in 3D space (i.e. not a circular motion).
	# Then the best history to pass into the FFT will be the
	# devices position (accelerometer reading) along that line,
	# i.e. a projection of the position vector onto the line.
	# Call that line the primary direction of motion.
	
	# To compute the primary direction of motion, we will use
	# my friend the Singular Value Decomposition, which is a
	# technique for dimensionality reduction (e.g. 3D to 1D).
	# We pass the SVD the history array, and it computes three
	# (because the original vectors are 3D) new vectors that
	# when linearly combined best approximate the original
	# matrix of history vectors. The new vectors are ordered,
	# and the first is the vector that explains most of the
	# original history matrix. Thus, it is the primary direction
	# of motion.
	
	# In the language of FFTs, there is a DC component to the
	# accelerometer data. At rest, the accelerometer reports the
	# force of gravity. We must subtract the DC component before
	# using SVD, or else it will get in the way.
	#
	# Subtract off the mean value on each axis from the history.
	m = [0, 0, 0]
	for i in xrange(3): m[i] = numpy.mean([h[i] for h in history_acc])
	m = numpy.array(m)
	history = [h-m for h in history_acc]
	
	# Compute the SVD.
	u, s, vT = numpy.linalg.svd(history)
	
	# vT[0]  - the primary direction of motion (vector norm is 1.0)
	# u[:,0] - the coordinate on the 'primary direction of motion'
	#          axis for each point in the history (vector norm is 1.0)
	# s[0]   - a scale factor that when applied to u and vT gives back
	#          the actual magnitudes of the acceleration over the history
	#          (so it is a sort of average absolute acceleration over
	#          the history).
	
	history_acc1d = u[:,0]
	
	# Compute the fundamental frequency of the 1D history data by passing
	# the 1D accelerometer values into a Fast Fourier Transform. This
	# gives a power level at a range of frequency components in the signal.
	FFT = scipy.fft(history_acc1d)
	
	# The FFT power values are complex numbers at first. Take abs to get
	# a real power. Also, weight low frequencies more highly since there
	# may be high-frequency energy especially at multiples of the fundamental
	# frequency. This is a made up exponential decay that seems to work well.
	FFTpower = abs(FFT)
	for i in xrange(len(FFT)):
		FFTpower[i] *= 2**(-i/float(len(history_acc1d)))
	
	# Find the peak of the power spectrum, and the frequency that corresponds
	# to that power.
	f0_i = numpy.argmax(FFTpower[1:]) + 1 # skip the DC component of the signal
	freqs = scipy.fftpack.fftfreq(len(history_acc1d), 1.0/sample_freq)
	f0 = abs(freqs[f0_i]) # FFT gives negative frequencies? Hmmm.
	
	# Add the computed frequency to a short history and smooth the frequency
	# by taking the mean over the history.
	history_f0.append(f0)
	history_f0.pop(0)
	f0 = numpy.mean(history_f0)
	
	# Print the fundamental frequency and the 'average' acceleration.
	print f0, s[0]
	
	# We can also predict the next beat by getting the phase of the
	# beat at the current time and projecting forward based on the
	# computed period, i.e. the inverse of the fundamental frequency.

	# This gets the current phase in radians in the range [-pi,pi].
	phase = numpy.angle(FFT[f0_i])
	
	# Predict the time of the next beat, in terms of the number of
	# samples from now.
	samples_per_beat = (sample_freq / f0)
	sample_of_next_beat = current_sample_num + (numpy.pi - phase)/(2*numpy.pi) * samples_per_beat
	
	# The phase is very wobbly over time, so we need to smooth it.
	# Add this prediction to a history, and get the predicted next
	# beat as the median. Exclude the current prediction because if
	# we are supposed to beat now, the current prediction might
	# jump to the next beat before we actually issue the beat.
	history_phase.append(sample_of_next_beat)
	history_phase.pop(0)
	sample_of_next_beat = numpy.median(history_phase[:-1])
	
	# Should we beat now? Only if we're past (but not too far past)
	# a predicted beat time, and if we're not too close to the
	# last beat.
	if sample_of_next_beat <= current_sample_num <= sample_of_next_beat+.2*samples_per_beat \
	  and current_sample_num > last_beat + .8*samples_per_beat:
		print "BEAT!"
		last_beat = current_sample_num
	
