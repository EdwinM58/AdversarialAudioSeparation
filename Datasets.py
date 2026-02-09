import numpy as np
from lxml import etree
import os.path
import librosa
import soundfile as sf

import Input.Input
from Sample import Sample


def subtract_audio(mix_list, instrument_list):
    '''
    Generates new audio by subtracting the audio signal of an instrument recording from a mixture
    :param mix_list: 
    :param instrument_list: 
    :return: 
    '''

    assert(len(mix_list) == len(instrument_list))
    new_audio_list = list()

    for i in range(0, len(mix_list)):
        new_audio_path = os.path.dirname(mix_list[i]) + os.path.sep + "remainingmix" + os.path.splitext(mix_list[i])[1]
        new_audio_list.append(new_audio_path)

        if os.path.exists(new_audio_path):
            continue
        mix_audio, mix_sr = librosa.load(mix_list[i], mono=False, sr=None)
        inst_audio, inst_sr = librosa.load(instrument_list[i], mono=False, sr=None)
        assert (mix_sr == inst_sr)
        new_audio = mix_audio - inst_audio
        if not (np.min(new_audio) >= -1.0 and np.max(new_audio) <= 1.0):
            print("Warning: Audio for mix " + str(new_audio_path) + " exceeds [-1,1] float range!")

        sf.write(new_audio_path, new_audio.T if new_audio.ndim > 1 else new_audio, mix_sr) #TODO switch to compressed writing
        print("Wrote accompaniment for song " + mix_list[i])
    return new_audio_list

def create_sample(db_path, instrument_node):
   path = db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text
   sample_rate = int(instrument_node.xpath("./sampleRate")[0].text)
   channels = int(instrument_node.xpath("./numChannels")[0].text)
   duration = float(instrument_node.xpath("./length")[0].text)
   return Sample(path, sample_rate, channels, duration)

def getDSDFilelist(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    train_vocals, test_vocals, train_mixes, test_mixes, train_accs, test_accs = list(), list(), list(), list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        vocals = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        [acc_path] = subtract_audio([mix.path], [vocals.path])
        acc = Sample(acc_path, vocals.sample_rate, vocals.channels, vocals.duration) # Accompaniment has same signal properties as vocals and mix

        if track.xpath("./databaseSplit")[0].text == "Training":
            train_vocals.append(vocals)
            train_mixes.append(mix)
            train_accs.append(acc)
        else:
            test_vocals.append(vocals)
            test_mixes.append(mix)
            test_accs.append(acc)

    return [train_mixes, train_accs, train_vocals], [test_mixes, test_accs, test_vocals]

def getCCMixter(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    mixes, accs, vocals = list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        voice = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        acc = create_sample(db_path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])

        mixes.append(mix)
        accs.append(acc)
        vocals.append(voice)

    return [mixes, accs, vocals]

def getIKala(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    mixes, accs, vocals = list(), list(), list()

    for track in tracks:
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        orig_path = mix.path
        mix_path = orig_path + "_mix.wav"
        acc_path = orig_path + "_acc.wav"
        voice_path = orig_path + "_voice.wav"

        mix_audio, mix_sr = librosa.load(mix.path, sr=None, mono=False)
        mix.path = mix_path
        sf.write(mix_path, np.sum(mix_audio, axis=0), mix_sr)
        sf.write(acc_path, mix_audio[0,:], mix_sr)
        sf.write(voice_path, mix_audio[1, :], mix_sr)

        voice = create_sample(mix.path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        voice.path = voice_path
        acc = create_sample(mix.path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])
        acc.path = acc_path

        mixes.append(mix)
        accs.append(acc)
        vocals.append(voice)

    return [mixes, accs, vocals]

def getMedleyDB(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text

    mixes, accs, vocals = list(), list(), list()

    tracks = root.xpath(".//track")
    for track in tracks:
        instrument_paths = list()
        # Mix together vocals, if they exist
        vocal_tracks = track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
                       track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
                       track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath")
        if len(vocal_tracks) > 0: # If there are vocals, get their file paths and mix them together
            vocal_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in vocal_tracks], "vocalmix")
            instrument_paths.append(vocal_track)
            vocals.append(Sample.from_path(vocal_track))
        else: # Otherwise append duration of track so silent input can be generated later on-the-fly
            duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
            vocals.append(duration)

        # Mix together accompaniment, if it exists
        acc_tracks = track.xpath(".//instrument[not(instrumentName='Voice') and not(instrumentName='Mix') and not(instrumentName='Instrumental')]/relativeFilepath") #TODO # We assume that there is no distinction between male/female here
        if len(acc_tracks) > 0:  # If there are vocals, get their file paths and mix them together
            acc_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in acc_tracks], "accmix")
            instrument_paths.append(acc_track)
            accs.append(Sample.from_path(acc_track))
        else:  # Otherwise append duration of track so silent input can be generated later on-the-fly
            duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
            accs.append(duration)

        # Mix together vocals and accompaniment
        mix_track = Input.Input.add_audio(instrument_paths, "totalmix")
        mixes.append(Sample.from_path(mix_track))

    return [mixes, accs, vocals]

def getFMA(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text

    mixes, accs, vocals = list(), list(), list()

    vocal_tracks = root.xpath(".//track/instrumentList/instrument[instrumentName='Mix']")
    instrumental_tracks = root.xpath(".//track/instrumentList/instrument[instrumentName='Instrumental']")
    for instr in vocal_tracks:
        mixes.append(create_sample(db_path,instr))

    for instr in instrumental_tracks:
        mixes.append(create_sample(db_path,instr))
        accs.append(create_sample(db_path,instr))

    return mixes, accs, vocals


def getMUSDB18(musdb_path, is_wav=False):
    '''
    Loads the MUSDB18 dataset using the musdb package.
    Exports stems to wav files on first run for fast subsequent loading.
    Switching from compressed to HQ: set is_wav=True and point musdb_path to the HQ folder.

    :param musdb_path: Path to the MUSDB18 root directory
    :param is_wav: Set True for MUSDB18-HQ (wav stems), False for compressed (.stem.mp4)
    :return: (train_list, test_list) each as [mixes, accs, vocals]
    '''
    import musdb

    mus = musdb.DB(root=musdb_path, is_wav=is_wav)
    tracks = mus.load_mus_tracks()

    stem_dir = os.path.join(musdb_path, "exported_stems")

    train_mixes, train_accs, train_vocals = list(), list(), list()
    test_mixes, test_accs, test_vocals = list(), list(), list()

    for track in tracks:
        track_dir = os.path.join(stem_dir, track.name)
        os.makedirs(track_dir, exist_ok=True)

        mix_path = os.path.join(track_dir, "mix.wav")
        vocals_path = os.path.join(track_dir, "vocals.wav")
        acc_path = os.path.join(track_dir, "accompaniment.wav")

        # Export stems to wav on first run
        if not os.path.exists(mix_path):
            print(f"Exporting stems for: {track.name}")
            sf.write(mix_path, track.audio, track.rate)
            sf.write(vocals_path, track.targets['vocals'].audio, track.rate)
            sf.write(acc_path, track.targets['accompaniment'].audio, track.rate)

        mix = Sample.from_path(mix_path)
        vocals = Sample.from_path(vocals_path)
        acc = Sample.from_path(acc_path)

        if track.subset == 'train':
            train_mixes.append(mix)
            train_accs.append(acc)
            train_vocals.append(vocals)
        else:
            test_mixes.append(mix)
            test_accs.append(acc)
            test_vocals.append(vocals)

    print(f"MUSDB18: {len(train_mixes)} train, {len(test_mixes)} test tracks loaded")
    return [train_mixes, train_accs, train_vocals], [test_mixes, test_accs, test_vocals]


def getMoisesDB(moisesdb_path):
    '''
    Loads the MoisesDB dataset.
    Combines vocal stems into a single vocals track and all other stems into accompaniment.
    Exports combined stems to wav files on first run.

    :param moisesdb_path: Path to the MoisesDB root directory
    :return: [mixes, accs, vocals] lists of Sample objects
    '''
    mixes, accs, vocals = list(), list(), list()

    # MoisesDB structure: each track is a folder containing stem wav files
    # with a hierarchy: track_name/stems/stem_name.wav
    tracks_dir = moisesdb_path
    if os.path.isdir(os.path.join(moisesdb_path, "tracks")):
        tracks_dir = os.path.join(moisesdb_path, "tracks")

    for track_name in sorted(os.listdir(tracks_dir)):
        track_path = os.path.join(tracks_dir, track_name)
        if not os.path.isdir(track_path):
            continue

        # Look for stems directory
        stems_path = track_path
        if os.path.isdir(os.path.join(track_path, "stems")):
            stems_path = os.path.join(track_path, "stems")

        # Find all stem wav files
        stem_files = [f for f in os.listdir(stems_path) if f.endswith('.wav')]
        if not stem_files:
            continue

        # Classify stems as vocal or non-vocal
        vocal_keywords = ['vocal', 'voice', 'sing', 'choir', 'backing_vocal', 'lead_vocal']
        vocal_stems = []
        acc_stems = []

        for stem_file in stem_files:
            stem_lower = stem_file.lower()
            if any(kw in stem_lower for kw in vocal_keywords):
                vocal_stems.append(os.path.join(stems_path, stem_file))
            else:
                acc_stems.append(os.path.join(stems_path, stem_file))

        if not vocal_stems and not acc_stems:
            continue

        # Export combined vocals, accompaniment, and mix
        export_dir = os.path.join(moisesdb_path, "exported_stems", track_name)
        os.makedirs(export_dir, exist_ok=True)

        mix_path = os.path.join(export_dir, "mix.wav")
        vocals_path = os.path.join(export_dir, "vocals.wav")
        acc_path = os.path.join(export_dir, "accompaniment.wav")

        if not os.path.exists(mix_path):
            print(f"Exporting stems for: {track_name}")
            all_audio = None
            vocal_audio = None
            acc_audio = None
            sr = None

            # Sum vocal stems
            for stem_path in vocal_stems:
                audio, file_sr = librosa.load(stem_path, mono=False, sr=None)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                sr = file_sr
                if vocal_audio is None:
                    vocal_audio = audio
                else:
                    min_len = min(vocal_audio.shape[1], audio.shape[1])
                    vocal_audio = vocal_audio[:, :min_len] + audio[:, :min_len]

            # Sum accompaniment stems
            for stem_path in acc_stems:
                audio, file_sr = librosa.load(stem_path, mono=False, sr=None)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                sr = file_sr
                if acc_audio is None:
                    acc_audio = audio
                else:
                    min_len = min(acc_audio.shape[1], audio.shape[1])
                    acc_audio = acc_audio[:, :min_len] + audio[:, :min_len]

            # Create mix = vocals + accompaniment
            if vocal_audio is not None and acc_audio is not None:
                min_len = min(vocal_audio.shape[1], acc_audio.shape[1])
                vocal_audio = vocal_audio[:, :min_len]
                acc_audio = acc_audio[:, :min_len]
                all_audio = vocal_audio + acc_audio
            elif vocal_audio is not None:
                all_audio = vocal_audio
                acc_audio = np.zeros_like(vocal_audio)
            elif acc_audio is not None:
                all_audio = acc_audio
                vocal_audio = np.zeros_like(acc_audio)

            sf.write(mix_path, all_audio.T if all_audio.ndim > 1 else all_audio, sr)
            sf.write(vocals_path, vocal_audio.T if vocal_audio.ndim > 1 else vocal_audio, sr)
            sf.write(acc_path, acc_audio.T if acc_audio.ndim > 1 else acc_audio, sr)

        mixes.append(Sample.from_path(mix_path))
        accs.append(Sample.from_path(acc_path))
        vocals.append(Sample.from_path(vocals_path))

    print(f"MoisesDB: {len(mixes)} tracks loaded")
    return [mixes, accs, vocals]