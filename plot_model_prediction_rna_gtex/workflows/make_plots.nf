include {MAKE_SCATTER} from '../modules/make_scatter.nf'

workflow MAKE_PLOTS {
    take:
    merged_dataset

    main:
    MAKE_SCATTER(merged_dataset)
    scatter_plots = MAKE_SCATTER.out.scatter_plots    

    emit:
    scatter_plots

}