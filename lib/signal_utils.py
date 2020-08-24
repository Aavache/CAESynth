# This code implementation was mainly taken from: https://github.com/ss12f32v/GANsynth-pytorch
# External Libraries
import numpy as np
import librosa
# Internal Libraries

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

# *********************************************************************#
# *********************SPECTROGRAM OPERATIONS *************************#
# *********************************************************************#

def mel_to_hertz(mel_values):
    """Converts frequencies in `mel_values` from the mel scale to linear scale."""
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)

def hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))

def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.
    Args:
        num_mel_bins: Int, number of output frequency dimensions.
        num_spectrogram_bins: Int, number of input frequency dimensions.
        sample_rate: Int, sample rate of the audio.
        lower_edge_hertz: Float, lowest frequency to consider.
        upper_edge_hertz: Float, highest frequency to consider.
    Returns:
        Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
    Raises:
        ValueError: Input argument in the wrong range.
    """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
            (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
            'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
            % (upper_edge_hertz, sample_rate))

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
            0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2)

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
    if upper_hz - lower_hz < freq_th:
        rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
        dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs**2))
        lower_edge_mel[i] = center_mel[i] - dm
        upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (
            center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (
            upper_edge_hz - center_hz)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]],
                'constant')
    return mel_weights_matrix

def _linear_to_mel_matrix():
    """Get the mel transformation matrix."""
    _sample_rate=16000
    _mel_downscale=1
    num_freq_bins = 2048 // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = 16000 / 2.0
    num_mel_bins = num_freq_bins // _mel_downscale
    return linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, _sample_rate, lower_edge_hertz,
        upper_edge_hertz)

def _mel_to_linear_matrix():
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def melspecgrams_to_specgrams(logmelmag2 = None, mel_p = None):
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    """
    mel2l = _mel_to_linear_matrix()
    logmag = None
    p = None
    if logmelmag2 is not None:
        logmelmag2 = logmelmag2.T
        logmelmag2 = np.array([logmelmag2])
        mag2 = np.tensordot(np.exp(logmelmag2), mel2l, 1)
        logmag = 0.5 * np.log(mag2+1e-6)
        logmag = logmag[0].T
    if mel_p is not None:
        mel_p = mel_p.T
        mel_p = np.array([mel_p])
        mel_phase_angle = np.cumsum(mel_p * np.pi, axis=1)
        phase_angle = np.tensordot(mel_phase_angle, mel2l, 1)
        p = instantaneous_frequency(phase_angle,time_axis=1)
        p = p[0].T
    return logmag, p


def specgrams_to_melspecgrams(magnitude, IF=None):
    """Converts specgrams to melspecgrams.
    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    """
    logmag = magnitude.T
    mag2 = np.exp(2.0 * logmag)
    mag2 = np.array([mag2])
    l2mel = _linear_to_mel_matrix()

    logmelmag2 = np.log(np.tensordot(mag2,l2mel,axes=1) + 1e-6)

    if IF is not None:
        p = IF.T
        phase_angle = np.cumsum(p * np.pi, axis=1)
        phase_angle = np.array([phase_angle])
        mel_phase_angle = np.tensordot(phase_angle, l2mel, axes=1)
        mel_p = instantaneous_frequency(mel_phase_angle,time_axis=1)
        return logmelmag2[0].T, mel_p[0].T
    else:
        return logmelmag2[0].T


# *********************************************************************#
# **************************PHASE OPERATIONS **************************#
# *********************************************************************#

def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1

    size = list(shape)
    size[axis] -= 1
    
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
#     print("dd",dd)
    ddmod = np.mod(dd+np.pi,2.0*np.pi)-np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
#     print("ddmod",ddmod)

    idx = np.logical_and(np.equal(ddmod, -np.pi),np.greater(dd,0)) # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
#     print("idx",idx)
    ddmod = np.where(idx, np.ones_like(ddmod) *np.pi, ddmod) # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
#     print("ddmod",ddmod)
    ph_correct = ddmod - dd
#     print("ph_corrct",ph_correct)
    
    idx = np.less(np.abs(dd), discont) # idx = tf.less(tf.abs(dd), discont)
    
    ddmod = np.where(idx, np.zeros_like(ddmod), dd) # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis) # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
#     print("idx",idx)
#     print("ddmod",ddmod)
#     print("ph_cumsum",ph_cumsum)
    
    
    shape = np.array(p.shape) # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis) 
    #ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
#     print("unwrapped",unwrapped)
    return unwrapped


def instantaneous_frequency(phase_angle, time_axis):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Args:
    phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
    time_axis: Axis over which to unwrap and take finite difference.
    Returns:
    dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
#     print("phase_unwrapped",phase_unwrapped.shape)
    
    dphase = diff(phase_unwrapped, axis=time_axis)
#     print("dphase",dphase.shape)
    
    # Add an initial phase to dphase
    size = np.array(phase_unwrapped.shape)
#     size = phase_unwrapped.get_shape().as_list()

    size[time_axis] = 1
#     print("size",size)
    begin = [0 for unused_s in size]
#     phase_slice = tf.slice(phase_unwrapped, begin, size)
#     print("begin",begin)
    phase_slice = phase_unwrapped[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1]]
#     print("phase_slice",phase_slice.shape)
    dphase = np.concatenate([phase_slice, dphase], axis=time_axis) / np.pi

#     dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
#     mag = np.complex(mag)
    temp_mag = np.zeros(mag.shape,dtype=np.complex_)
    temp_phase = np.zeros(mag.shape,dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
#             print(mag[i,j])
            temp_mag[i,j] = np.complex(mag[i,j])
#             print(temp_mag[i,j])
    
    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i,j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))
#             print(temp_mag[i,j])
    
#     phase = np.complex(np.cos(phase_angle), np.sin(phase_angle))
   
    return temp_mag * temp_phase