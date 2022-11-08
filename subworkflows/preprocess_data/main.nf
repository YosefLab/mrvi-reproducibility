include { preprocess_data as preprocess } from params.module.preprocess_data


workflow preprocess_data {
    take:
    input

    main:
    preprocess(input)

    emit:
    preprocess.out
}
