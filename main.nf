#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

def helloWorld() {
  println "hello world"
}

workflow {
  helloWorld()
}
