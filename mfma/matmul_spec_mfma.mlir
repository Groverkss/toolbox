// A: vector<64x32xf16>
// B: vector<64x32xf16> 
// C: vector<64x64xf32>
// This depends on the tile sizes used.

// A: vector<64x32xf16>
#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [2, 32]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout_a = #iree_vector_ext.layout<#row_layout, #col_layout>

// B: vector<64x32xf16> 
#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [2, 32]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout_b = #iree_vector_ext.layout<#row_layout1, #col_layout1>

// C: vector<64x64xf32>
#row_layout2 = #iree_vector_ext.per_dim_layout<[BATCHX, VECTORY, LANEY, VECTORX], [2, 4, 2, 4]>
#col_layout2 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [2, 32]>
#layout_c = #iree_vector_ext.layout<#row_layout2, #col_layout2>

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // TODO: This is obviously not matching a matmul and is just a hack.
    transform.match.operation_name %entry ["linalg.generic"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @codegen(%variant_op: !transform.any_op {transform.consumed}) {
    // Get matmul op
    // ==========================================
    %matmul = transform.collect_matching @match_matmul in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_matmul, %forall_grid  = transform.structured.tile_using_forall %matmul tile_sizes [256, 128]
    ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Fuse fill
    // ==========================================
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    transform.structured.fuse_into_containing_op %fill into %forall_grid :
    (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func0 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Tile fill
    // ==========================================
    %fill2 = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %tiled_fill3, %forall3 = transform.structured.tile_using_forall %fill2 tile_sizes [64, 64] (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile reduction dim
    // ==========================================
    %tiled_matmul2, %loop = transform.structured.tile_using_for %tiled_matmul [0, 0, 32] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote lhs and rhs
    // ==========================================
    %promoted_matmul, %alloc_a, %alloc_b = transform.iree.promote_operands %tiled_matmul2 [0, 1]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile to warps
    // ==========================================
    %tiled_matmul3, %forall2 = transform.structured.tile_using_forall %promoted_matmul tile_sizes [64, 64] (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.apply_cse to %func0 : !transform.any_op

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 8, 1] subgroup_size = 64 sync_after_distribution = true : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Get the vector.contract op.
    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op_3 :  (!transform.any_op) -> !transform.any_op

    // Step 7. SIMD -> SIMT Using layouts
    // ===========================================================================
    %layoutA = transform.param.constant #layout_a -> !transform.any_param
    %layoutB = transform.param.constant #layout_b -> !transform.any_param
    %layoutC = transform.param.constant #layout_c -> !transform.any_param

    transform.annotate %contract "__vector_layout_test_anchor_operand_0" = %layoutA : !transform.any_op, !transform.any_param
    transform.annotate %contract "__vector_layout_test_anchor_operand_1" = %layoutB : !transform.any_op, !transform.any_param
    transform.annotate %contract "__vector_layout_test_anchor_operand_2" = %layoutC : !transform.any_op, !transform.any_param

    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    transform.iree.test_amdgpu_contraction_distribution %top_level_func : !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    transform.print %variant_op_3 : !transform.any_op

    transform.yield
  }
}
