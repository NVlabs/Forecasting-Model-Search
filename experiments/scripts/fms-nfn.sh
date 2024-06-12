# on one-cnn-benchmark
python fms.py --experiment_type fms-nfn --park simple_cnn_park
python fms.py --experiment_type fms-nfn --park simple_cnn_park --ablate-cnn

# won't work on pretrained_model_park - nfn can't process diverse architectures (though UNFs and the GMN can!)
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_cifar10
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_cifar10 --ablate-cnn
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_svhn
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_svhn --ablate-cnn

# doesn't make sense to assess transfer performance for the same reasons; running on simple-cnn-park is the extent of transferability
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_transfer_cifar10
# python fms.py --experiment_type fms-nfn --park pretrained_model_park_transfer_svhn
# python fms.py --experiment_type fms-nfn --park simple_cnn_park_transfer