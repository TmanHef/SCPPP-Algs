(define (problem rovers-3-3-success-0-1-rover1) (:domain rovers-3-3-success-0-1-rover1)
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
		rover1 - rover
	)
)
(:init

    ;; private
	(store_of rover1soilStore rover1)
	(equipped_for_soil_analysis rover1)
	(can_traverse rover1 base quarry)
	(can_traverse rover1 quarry base)
	(can_traverse rover1 base dune)
	(can_traverse rover1 dune base)
	(at rover1 base)

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