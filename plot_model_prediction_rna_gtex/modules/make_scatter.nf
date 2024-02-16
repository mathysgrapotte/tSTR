process MAKE_SCATTER {

    input:
    path(merged_file)

    output:
    path("scatter_plot_*"), emit: scatter_plots

    script:
    """
    launch_make_plots.py -i ${merged_file} -o scatter_plot
    """

}