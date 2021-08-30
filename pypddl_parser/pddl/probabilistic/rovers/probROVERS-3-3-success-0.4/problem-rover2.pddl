(define (problem rovers-3-3-success-0-4-rover2) (:domain rovers-3-3-success-0-4-rover2)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mars - objective
    mothership - lander
    rover0RockStore - store
    rover1soilStore - store
    rover2RockStore - store
    rover2soilStore - store
    rover2cam - camera
    color - mode


	(:private
		rover2 - rover
	)
)
(:init

    ;; private
	(store_of rover2RockStore rover2)
	(store_of rover2soilStore rover2)
	(equipped_for_rock_analysis rover2)
	(equipped_for_soil_analysis rover2)
	(equipped_for_imaging rover2)
	(can_traverse rover2 base quarry)
	(can_traverse rover2 quarry base)
	(can_traverse rover2 base dune)
	(can_traverse rover2 dune base)
	(on_board rover2cam rover2)
	(at rover2 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(visible_from mars dune)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover1soilStore)
	(empty rover2RockStore)
	(empty rover2soilStore)
	(supports rover2cam color)
	(calibration_target rover2cam mars)

)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
		(communicated_image_data mars color)
	)
)
)