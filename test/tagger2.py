from pyAudioAnalysis import audioTrainTest as aT

c, p, p_nam = aT.file_classification("music/ambient-classical-guitar-144998.wav", "data/models/knn_musical_genre_6","knn")
for k in range(len(p_nam)):
    print(f'P({p_nam[k]}={p[k]})')