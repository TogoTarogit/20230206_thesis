import numpy as np
import mne


# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
#                         'sample_audvis_filt-0-40_raw.fif')
# raw = mne.io.read_raw_fif(sample_data_raw_file)

# # print("ready")
# # print(raw)

# # raw.plot_psd(fmax=50)
# # raw.plot(duration =5,n_channels = 30)

# # set up and fit the ICA
# ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
# ica.fit(raw)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=ica.exclude)

# orig_raw = raw.copy()
# raw.load_data()
# ica.apply(raw)

# # show some frontal channels to clearly illustrate the artifact removal
# chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
#        'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
#        'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
#        'EEG 007', 'EEG 008']
# chan_idxs = [raw.ch_names.index(ch) for ch in chs]
# orig_raw.plot(order=chan_idxs, start=12, duration=4)
# raw.plot(order=chan_idxs, start=12, duration=4)

# events = mne.find_events(raw, stim_channel='STI 014')
# print(events[:5])  # show the first 5

# reject_criteria = dict(mag=4000e-15,     # 4000 fT
#                        grad=4000e-13,    # 4000 fT/cm
#                        eeg=150e-6,       # 150 µV
#                        eog=250e-6)       # 250 µV

# epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
#                     reject=reject_criteria, preload=True)


csv_path = r""
my_anno = mne.read_annotations(csv_path)







print("finish func")
