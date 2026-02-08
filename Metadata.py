import os, re, subprocess
from soundfile import SoundFile

_find_sampling_rate = re.compile('.* ([0-9:]+) Hz,', re.MULTILINE )
_find_channels = re.compile(".*Hz,( .*?),", re.MULTILINE)
_find_duration = re.compile('.*Duration: ([0-9:]+)', re.MULTILINE )

def timestamp_to_seconds( ms ):
    "Convert a hours:minutes:seconds string representation to the appropriate time in seconds."
    a = ms.split(':')
    assert 3 == len( a )
    return float(a[0]) * 3600 + float(a[1]) * 60 + float(a[2])

def seconds_to_min_sec( secs ):
    "Return a minutes:seconds string representation of the given number of seconds."
    mins = int(secs) // 60
    secs = int(secs - (mins * 60))
    return f"{mins}:{secs:02d}"

def get_mp3_metadata(audio_path):
    "Determine length of tracks listed in the given input files (e.g. playlists)."
    try:
        ffmpeg = subprocess.check_output(
          ['ffmpeg', '-i', audio_path],
          stderr=subprocess.STDOUT )
    except subprocess.CalledProcessError as e:
        ffmpeg = e.output

    ffmpeg = ffmpeg.decode('utf-8', errors='replace')

    # Get sampling rate
    match = _find_sampling_rate.search( ffmpeg )
    assert(match)
    sampling_rate = int(match.group( 1 ))

    # Get channels
    match = _find_channels.search( ffmpeg )
    assert(match)
    channels = match.group( 1 )
    channels = (2 if "stereo" in channels else 1)

    # Get duration
    match = _find_duration.search( ffmpeg )
    assert(match)
    duration = match.group( 1 )
    duration = timestamp_to_seconds(duration)

    return sampling_rate, channels, duration

def get_audio_metadata(audioPath, sphereType=False):
    '''
    Returns sampling rate, number of channels and duration of an audio file
    :param audioPath:
    :param sphereType:
    :return:
    '''
    ext = os.path.splitext(audioPath)[1][1:].lower()
    if ext=="aiff" or sphereType:  # AIFF/SPHERE headers
        snd_file = SoundFile(audioPath, mode='r')
        inf = snd_file._info
        sr = inf.samplerate
        channels = inf.channels
        duration = float(inf.frames) / float(inf.samplerate)
    elif ext=="mp3": # Use ffmpeg/ffprobe
        sr, channels, duration = get_mp3_metadata(audioPath)
    else:
        snd_file = SoundFile(audioPath, mode='r')
        inf = snd_file._info
        sr = inf.samplerate
        channels = inf.channels
        duration = float(inf.frames) / float(inf.samplerate)
    return int(sr), int(channels), float(duration)
