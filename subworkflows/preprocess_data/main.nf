include { preprocess } from params.modules.preprocess


workflow preprocess_data {
    take:
    inputs

    main:
    preprocess(inputs)

    emit:
    preprocess.out
}
