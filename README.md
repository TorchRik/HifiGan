
### Воспроизводимость:

Первым делом необходимо настроить новое окружение и установить зависимости, с помощью conda это можно сделать так

```bash
conda create --name hifi_gan_env python=3.12
conda activate hifi_gan_env

pip install -r requirements.txt
```

#### Inference:
Для запуска модели на новых текстовых данных необходимо запустить два скрипта:


```bash
python generate_audio.py datasets.custom_dataset.dataset_path = 'PATH_TO_DATA' \
  checkpoint_path: 'PATH_TO_CHECKPOINT' \
  path_to_save: 'PATH_TO_SAVE_DATA'
```

PATH_TO_DATA - Путь до папки с кастомными данными(должна содержать папку
transcriptions с текстовыми транскрипциями).

PATH_TO_CHECKPOINT - путь до чекпоинта, пример 'saved/generator-checkpoint-epoch70.pth'

PATH_TO_SAVE - путь до сохранения данных, файлы будут лежать в папке PATH_TO_SAVE


Для подсчета метрик необходимо запустить:

```bash
python score_generated.py path_to_score='PATH_TO_SAVED_AUDIO'
```
PATH_TO_SAVED_AUDIO - такой же, как PATH_TO_SAVE.

##### Training
Для воспроизведения обучения указать в configs/datasets/ljspeech путь до папки с датасетом, и затем запустить скрипт обучения:

```
python train.py datasets.train.dataset_path='PATH_TO_TRAIN_DATASET' \
datasets.val.dataset_path='PATH_TO_VAL_DATASET'
```

Example:

PATH_TO_TRAIN_DATASET = 'data/lj_speech_train'

PATH_TO_VAL_DATASET = 'data/lj_speech_test'

### Эксперименты:

#### Начальный сетап

В начале я пытался обучаться на семплах длительностью 10 секунд, однако спустя ночь обучения я заметил, что обучение так идет слишком медленно. И поскольку время на обучение было сильно ограничено решил попробовать сократить запись до 3 секунд.

Первый этап обучения можно посмотреть здесь:
https://wandb.ai/torchrik/HiFiGan/runs/3v6fh6ao?nw=nwusertorchrik


![[Screenshot 2024-12-08 at 18.30.40.png]]
![[Screenshot 2024-12-08 at 18.30.56.png]]

#### Изменения и итоговый сетап


Дальше все эксперименты проводились с аудиозаписями обрезанными до 3 секунд, запуски можно посмотреть здесь:
https://wandb.ai/torchrik/HiFiGan/runs/bluykepi?nw=nwusertorchrik
https://wandb.ai/torchrik/HiFiGan/runs/v1004qlw?nw=nwusertorchrik
https://wandb.ai/torchrik/HiFiGan/runs/jmw5z50w?nw=nwusertorchrik

Я пробовал разные размеры батча и в итоге остановился на размере в 32.
К сожалению за счет смен размеров батча немного сдвинулись графики(т.к от батча изменялся размер эпохи и как следствие логирование сдвигалось.)
#### Итог экспериментов и обучения

Я не заметил каких либо проблем во время обучения, кроме того, что сам процесс обучения занимает много времени. Для этой задаче довольно сложно достичь переобучения, и стандартный сетап обучения, предложенный авторами статьи работает без проблем, так что я сосредоточился на ускорении обучения.
В моем случае сильнее всего помогло сокращение длины аудиозаписи с 10 до 3 секунд, при этом наиболее эффективно показала себя тренировка с batch_size=32 при меньшем батче GPU утилизируется далеко не полностью, а при большем возрастает время итерации(что критично в условиях малого времени).

Так же стоит отметить, что достаточно быстро - уже после нескольких часов обучения, модель начинает генерировать записи, на которых можно хорошо различить слова, однако дальнейшее улучшение и уменьшение робастности занимает много времени.

##### Трудности
Основные трудности были перед запуском обучения модели - так для корректной работы алгоритма необходимо, что бы оригинальная wav и сгенерированная были одного размера. Я решил падить вавки до длины кратной 256(что бы преобразования audio -> mel spec -> audio) сохраняло размер, однако почему то преобразование даже обрезанной аудио в мел спектрограмму увеличивало длину последней на 1, так что в итоге решил просто падить исходную вав и спектрограмму до размеров сгенерированных. Тогда же поймал несколько наприятных багов в дискриминаторе (из-за использования += к вектору ломался торч, буду теперь знать).

Но в итоге из-за сложностей в процессе дебага запуска забыл поменять начальный размер генератора с 128 до 512, как следствие вероятно качество стало несколько хуже.

#### Inner-Analysis:

Сравним мел спектрограммы у оригинального аудио и сгенерированного:

В начале обучения:

Real spectrogram:
![[media_images_real_spectrogram_train_199_9f73ed8500ce6a49a034.png]]

Generated spectrogram:
![[media_images_generated_spectrogram_train_199_fdb0d3d22cc214498142 1.png]]

В конце обучения:

Real spectrogram:
![[media_images_real_spectrogram_train_train_28004_e2e5b040ef4c390e70a5.png]]

Spectrogram for generated audio:
![[Screenshot 2024-12-08 at 20.13.05.png]]


В начале обучения спектрограммы сильно отличаются и слабо выравнены по времени и частоте, что в целом логично.

При этом в конце обучения спектрограммы стали похожи, они выравнены и по времени и по частотам, однако спектрограмма реального аудио имеет более четко выделенные области(вероятно как раз сам голос).
При этом на этой стадии обучения все еще слышна робастность голоса, что вероятно и соотвествует более размытой спектрограмме.

Сравним теперь аудио представления:

Real audio:
![[Screenshot 2024-12-08 at 20.58.25.png]]

Generated audio:
![[Screenshot 2024-12-08 at 20.58.59.png]]

Видно, что дорожки выровнены по времени, но вместе с тем достаточно сильно отличаются по амплитуде. Так что и визуально и по самой дорожке можно определить реальную запись.

Так же я запустил WVMOS на сгенерированных аудио тестовой части датасета, значение метрики получилось 2.3(это немного, но честная работа). Вероятно не хватило времени на дообучения модели, по этому качество не идеальное.

**External Dataset Analysis**:

Я сгенерировал 5 аудизаписей из текста в часте Grade, и измерил WVMOS на них - получил 1.6. Видно, что хуже, чем на данных из датасета. Вероятно это связано с тем, что даныне пришли из немного другого распределния.



**Full-TTS system Analysis**:

Я сгенерировал аудиозаписи из текстовых данных(используя FastSpeech2)
сами ауидо приложил в анитаске, WVMOS получился равным 1.58.
